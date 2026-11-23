"""Frozen scoring tasks; organ metadata stays outside the network interface."""
from __future__ import annotations

from common import COUNTS, MARKERS, ORGANS, WEIGHTS
from dataio import load_index

TASK_MARKERS = {'colon': 'CD68', 'liver': 'CD45RO', 'stomach': 'CD68'}
MARKER_IDS = {'CD68': 0, 'CD45RO': 1}
CANDIDATE = 'M07'


def validate_tasks(records):
    if not records:
        raise ValueError('Empty M07 task list')
    for row in records:
        marker = row.get('marker')
        if row.get('organ') not in TASK_MARKERS or marker != TASK_MARKERS[row['organ']]:
            raise ValueError('M07 organ/Marker task mismatch')
        if row.get('subset') not in ('train', 'val'):
            raise ValueError('M07 currently accepts Train/Val only')
        if row.get('target') and '/'+marker+'/' not in row['target']:
            raise ValueError('M07 target channel does not match requested Marker')


def load_tasks(path, subset):
    if subset not in ('train', 'val'):
        raise ValueError('M07 task loading is Train/Val only')
    records = load_index(path, subset=subset)
    validate_tasks(records)
    return records


def task_exposures(step):
    return {organ+'/'+TASK_MARKERS[organ]: step*COUNTS[organ] for organ in ORGANS}


def validate_recipe(cfg):
    if (cfg.get('id') != 'semi_m07_r1' or cfg.get('candidate') != CANDIDATE
            or cfg.get('method') != 'M07-B' or cfg.get('markers') != list(MARKERS)
            or cfg.get('task_markers') != TASK_MARKERS or cfg.get('marker_ids') != MARKER_IDS
            or cfg.get('initialization') != 'gaussian' or cfg.get('seed') != 2026
            or cfg.get('native_size') != 256 or cfg.get('microbatch') != 1
            or cfg.get('images_per_update') != COUNTS or cfg.get('domain_loss_weights') != WEIGHTS):
        raise ValueError('M07 frozen task/initialization contract mismatch')
    if (cfg.get('fixed_lr') != [dict(through_update=250, lr=1e-5), dict(through_update=500, lr=5e-5),
                              dict(through_update=8000, lr=1e-4), dict(through_update=10000, lr=5e-5)]
            or cfg.get('lr_schedule') != 'fixed' or cfg['early_stop']['enabled']
            or cfg.get('max_updates') != 10000 or cfg.get('soft_wall_seconds') != 28800
            or cfg.get('gradient_clip') != 1.0
            or cfg.get('film') != dict(locations=['latent', 'refinement'], initialization='identity')):
        raise ValueError('M07 schedule/conditioning contract mismatch')
