"""Single-image train/evaluate/infer primitives and exact resume state."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import (COUNTS, MARKERS, ORGANS, ROOT, DATA_SHA, WEIGHT_SHA,
                    fingerprint, sha256, write_json)
from dataio import paired_augmentation
from losses import combined_loss
from metrics import aggregate, image_metrics, png_pixels
from sampling import domain_coefficients


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    torch.save(payload, temp)
    os.replace(temp, path)


def save_snapshot(path, model, optimizer, scheduler, sampler, progress, contract):
    atomic_torch_save(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), progress=progress,
        contract=contract, rng=rng_state()))


def restore_snapshot(path, model, optimizer, scheduler, sampler, contract):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['contract'] != contract:
        raise ValueError('Checkpoint data/source/config contract changed')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    sampler.load_state_dict(payload['sampler'])
    restore_rng(payload['rng'])
    return payload['progress']


class TrainingNumericsError(FloatingPointError):
    def __init__(self, kind, **details):
        self.details = dict(kind=kind, **details)
        super().__init__(kind+'; optimizer update refused')


def check_gradient_elements(model):
    named = [(name, parameter.grad) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    if not named:
        raise TrainingNumericsError('missing_gradients')
    finite = torch.stack([torch.isfinite(grad).all() for _, grad in named]).cpu().tolist()
    bad = [name for (name, _), valid in zip(named, finite) if not valid]
    if bad:
        raise TrainingNumericsError('nonfinite_gradient_elements', bad_parameters=bad,
                                    all_gradient_elements_finite=False)
    return named


def checked_clip_gradients(model, clip_norm):
    named = check_gradient_elements(model)
    try:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if 'non-finite' not in str(error):
            raise
        # Diagnostic only: never replace the training clip or step after failure.
        norms = torch.stack([torch.linalg.vector_norm(grad.detach().double()) for _, grad in named])
        largest = sorted(((float(grad.detach().abs().max().cpu()), name) for name, grad in named),
                         reverse=True)[:10]
        raise TrainingNumericsError('nonfinite_total_norm', all_gradient_elements_finite=True,
            diagnostic_norm_fp64=float(torch.linalg.vector_norm(norms).cpu()),
            largest_absolute_gradients=[dict(parameter=name, maximum=value) for value, name in largest],
            original_error=str(error)) from error


def forward_record(model, image, row):
    if getattr(model, 'marker_conditioned', False):
        from tasks import MARKER_IDS
        if row is None or row.get('marker') not in MARKER_IDS:
            raise ValueError('An explicit M07 Marker task is required')
        return model(image, MARKER_IDS[row['marker']])
    return model(image)


def train_update(model, optimizer, reader, batch, step, device,
                 loss_fn=combined_loss, mixed_precision=True, clip_norm=1.0, failure_path=None):
    if Counter(row['organ'] for row in batch) != COUNTS:
        raise ValueError('An update requires exactly C1/L2/S5')
    if any(row['subset'] != 'train' for row in batch):
        raise ValueError('Held-out rows cannot enter training')
    if getattr(model, 'marker_conditioned', False):
        model.validate_tasks(batch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    coefficients = domain_coefficients()
    totals = {organ: 0.0 for organ in ORGANS}
    context = dict(update=step, successful_updates_before_failure=step-1, microbatch=0,
                   phase='input', samples=[dict(row) for row in batch], optimizer_step_performed=False,
                   optimizer_step_started=False, mixed_precision=mixed_precision,
                   optimizer_defaults=dict(optimizer.defaults),
                   augmentation='sha256(seed2026|update|position0|stem), rot90 then horizontal flip')
    entry_rng = rng_state()
    entry_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    micro_rng = []
    try:
        for position, row in enumerate(batch):
            micro_rng.append(rng_state())
            context.update(microbatch=position+1, sample=dict(row), phase='input')
            x, y = reader.pair(row)
            x, y = paired_augmentation(x, y, step, position, row['stem'])
            x = torch.from_numpy(x[None]).to(device)
            y = torch.from_numpy(y[None]).to(device)
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise TrainingNumericsError('nonfinite_input_or_target')
            context['phase'] = 'forward'
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=mixed_precision):
                prediction = forward_record(model, x, row)
            if prediction.shape != y.shape:
                raise TrainingNumericsError('invalid_prediction_shape', prediction_shape=list(prediction.shape),
                                            target_shape=list(y.shape))
            if not bool(torch.isfinite(prediction).all()):
                raise TrainingNumericsError('nonfinite_prediction')
            context['phase'] = 'loss'
            loss = loss_fn(prediction, y)
            if not bool(torch.isfinite(loss)):
                raise TrainingNumericsError('nonfinite_loss')
            context.update(phase='backward', loss=float(loss.detach().cpu()))
            (loss*coefficients[row['organ']]).backward()
            context['phase'] = 'gradient_after_microbatch'
            check_gradient_elements(model)
            totals[row['organ']] += float(loss.detach().cpu())/COUNTS[row['organ']]
        context['phase'] = 'gradient_after_accumulation'
        if hasattr(model, 'validate_gradients'):
            from tasks import MARKER_IDS
            model.validate_gradients({MARKER_IDS[r['marker']] for r in batch})
        norm = checked_clip_gradients(model, clip_norm)
        context['phase'] = 'optimizer'
        context['optimizer_step_started'] = True
        optimizer.step()
        context['optimizer_step_performed'] = True
        if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
            raise TrainingNumericsError('nonfinite_parameter_after_optimizer')
    except Exception as error:
        if failure_path is not None:
            record = dict(context, **getattr(error, 'details', {}),
                error=type(error).__name__+': '+str(error),
                interpretation='runtime failure, not a Marker performance result')
            try:
                write_json(failure_path, record)
                # Backward does not change Restormer parameters. Save only on failure.
                state = dict(model.state_dict())
                is_entry = not context['optimizer_step_started']
                if is_entry:
                    state.update(entry_buffers)
                folder = Path(failure_path).parent
                atomic_torch_save(folder/'failure_state.pt', dict(
                    purpose='no-step numerical replay only', model=state, rng=entry_rng,
                    microbatch_rng=micro_rng, samples=context['samples'], update=step,
                    mixed_precision=mixed_precision, model_state_is_update_entry=is_entry,
                    failure=record, model_class=type(model).__name__,
                    contract=dict(recipe_sha256=sha256(ROOT/'recipe.json'),
                        release_sha256=sha256(ROOT/'MANIFEST.sha256') if (ROOT/'MANIFEST.sha256').exists() else None,
                        official_sha256=DATA_SHA, public_sha256=WEIGHT_SHA)))
                atomic_torch_save(folder/'failure_optimizer.pt', optimizer.state_dict())
                record['diagnostic_state'] = 'failure_state.pt'
                record['diagnostic_optimizer'] = 'failure_optimizer.pt'
            except Exception as save_error:
                record['diagnostic_save_error'] = type(save_error).__name__+': '+str(save_error)
            try:
                write_json(failure_path, record)
            except OSError:
                # Disk failure must not replace the original numerical exception.
                import sys
                print('Could not persist numerical diagnostic: '+str(record), file=sys.stderr, flush=True)
        raise
    return dict(losses=totals, gradient_norm=float(norm.detach().cpu()))


@contextlib.contextmanager
def evaluation_context(model):
    state, mode, digest = rng_state(), model.training, model_digest(model)
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        restore_rng(state)
        model.train(mode)
        if model_digest(model) != digest:
            raise RuntimeError('Evaluation changed model persistent state')


def single_prediction(model, x, device, row=None):
    seed_all(2026)
    tensor = torch.from_numpy(np.array(x[None], copy=True)).to(device)
    prediction = forward_record(model, tensor, row).float().detach().cpu().numpy()
    if prediction.shape != (1, 1, 256, 256):
        raise ValueError('Invalid single-target output shape')
    return prediction[0, 0]


def evaluate(model, reader, records, device, csv_path, expected_subset='val'):
    if expected_subset not in ('train', 'val') or not records or any(row['subset'] != expected_subset for row in records):
        raise ValueError('Evaluation rows do not match the explicit Train/Val role')
    if getattr(model, 'marker_conditioned', False):
        model.validate_tasks(records)
    results = []
    with evaluation_context(model):
        for row in records:
            x, y = reader.pair(row)
            _, prediction = png_pixels(single_prediction(model, x, device, row))
            target = np.repeat(np.rint(y[0]*255).astype(np.uint8)[..., None], 3, axis=-1)
            result = dict(organ=row['organ'], roi=row['roi'], stem=row['stem'],
                          marker=row.get('marker', ''),
                          target_mean=float(target.mean()),
                          **image_metrics(target, prediction))
            results.append(result)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    result = aggregate(results)
    if getattr(model, 'marker_conditioned', False):
        from tasks import TASK_MARKERS
        result['task_markers'] = dict(TASK_MARKERS)
    result['evaluation_role'] = expected_subset
    return result


def predict_archive(model, reader, records, device, output, marker):
    if getattr(model, 'marker_conditioned', False):
        raise ValueError('M07 formal Test generation is not implemented in this Train/Val stage')
    if marker not in MARKERS or any(row['subset'] != 'test' for row in records):
        raise ValueError('Explicit candidate and official Test inputs required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite Test predictions')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.partial')
    if temporary.exists():
        raise FileExistsError('Existing partial inference output requires inspection')
    names = set()
    with evaluation_context(model), zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED) as archive:
        for row in records:
            if row['organ'] not in ORGANS or Path(row['stem']).name != row['stem']:
                raise ValueError('Unsafe output sample identity')
            name = 'results/%s/test/%s/%s_fake.png' % (row['organ'], marker, row['stem'])
            if name in names:
                raise ValueError('Duplicate prediction path')
            names.add(name)
            encoded, _ = png_pixels(single_prediction(model, reader.input(row), device))
            archive.writestr(name, encoded)
    os.replace(temporary, output)
