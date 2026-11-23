"""Reproducible 1/2/5 data tasks, accumulated into one optimizer update."""
from __future__ import annotations

import copy
import hashlib
import random
from collections import defaultdict

from common import COUNTS, ORGANS, WEIGHTS, fingerprint


def domain_coefficients():
    return {organ: WEIGHTS[organ]/COUNTS[organ] for organ in ORGANS}


class Cycle:
    def __init__(self, items, seed):
        if not items:
            raise ValueError('Empty data cycle')
        self.items = list(items)
        self.rng = random.Random(seed)
        self.rng.shuffle(self.items)
        self.cursor = 0
        self.pending = []
        self.draws = 0

    def take(self):
        self.draws += 1
        if self.pending:
            return self.pending.pop(0)
        if self.cursor == len(self.items):
            self.rng.shuffle(self.items)
            self.cursor = 0
        item = self.items[self.cursor]
        self.cursor += 1
        return item

    def unique(self, count):
        if count > len(self.items):
            raise ValueError('Too few distinct ROIs')
        selected, deferred = [], []
        while len(selected) < count:
            item = self.take()
            if item in selected:
                deferred.append(item)
            else:
                selected.append(item)
        self.pending = deferred + self.pending
        return selected

    def state_dict(self):
        return {'items': self.items[:], 'cursor': self.cursor, 'pending': self.pending[:],
                'draws': self.draws, 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if sorted(state['items']) != sorted(self.items):
            raise ValueError('Cycle population changed')
        self.items = state['items'][:]
        self.cursor = state['cursor']
        self.pending = state['pending'][:]
        self.draws = state['draws']
        self.rng.setstate(state['rng'])


class DomainSampler:
    def __init__(self, records, seed=2026):
        self.records = [dict(row) for row in records]
        if any(row['subset'] != 'train' for row in self.records):
            raise ValueError('Training sampler received held-out images')
        self.identity = fingerprint(self.records)
        self.cycles = {}
        def make(key, items):
            derived = int.from_bytes(hashlib.sha256(('%d|%s' % (seed, key)).encode()).digest()[:8], 'big')
            self.cycles[key] = Cycle(items, derived)
        self.groups = {}
        for organ in ORGANS:
            grouped = defaultdict(list)
            for i, row in enumerate(self.records):
                if row['organ'] == organ:
                    grouped[row['roi']].append(i)
            if not grouped:
                raise ValueError('Missing domain: ' + organ)
            self.groups[organ] = dict(grouped)
            if organ == 'colon':
                make(organ, sorted(i for group in grouped.values() for i in group))
            else:
                if len({len(group) for group in grouped.values()}) != 1:
                    raise ValueError('ROI-balanced sampling requires equal ROI sizes')
                make(organ, sorted(grouped))
                for roi in sorted(grouped):
                    make(organ + '/' + roi, grouped[roi])
        self.updates = 0

    def next_update(self):
        selected = []
        for organ in ORGANS:
            if organ == 'colon':
                selected.append(self.records[self.cycles[organ].take()])
            else:
                for roi in self.cycles[organ].unique(COUNTS[organ]):
                    selected.append(self.records[self.cycles[organ + '/' + roi].take()])
        self.updates += 1
        return selected

    def state_dict(self):
        return {'identity': self.identity, 'updates': self.updates,
                'cycles': {key: cycle.state_dict() for key, cycle in self.cycles.items()}}

    def load_state_dict(self, state):
        if state['identity'] != self.identity or set(state['cycles']) != set(self.cycles):
            raise ValueError('Sampler data/structure changed')
        for key, value in state['cycles'].items():
            self.cycles[key].load_state_dict(copy.deepcopy(value))
        self.updates = state['updates']


def train_evaluation_panel(records, seed=2026, per_organ=140):
    if not records or per_organ < 1 or any(r['subset'] != 'train' for r in records):
        raise ValueError('Diagnostic panel must contain training rows only')
    panel = []
    for organ in ORGANS:
        groups = defaultdict(list)
        for row in records:
            if row['organ'] == organ:
                groups[row['roi']].append(dict(row))
        if not groups:
            raise ValueError('Diagnostic panel is missing an organ')
        queues = []
        for roi in sorted(groups):
            items = sorted(groups[roi], key=lambda r: r['stem'])
            derived = int.from_bytes(hashlib.sha256(('%d|train-eval|%s|%s' %
                                    (seed, organ, roi)).encode()).digest()[:8], 'big')
            random.Random(derived).shuffle(items)
            queues.append(items)
        count = min(per_organ, sum(len(q) for q in queues))
        for offset in range(max(len(q) for q in queues)):
            for queue in queues:
                if offset < len(queue) and count:
                    panel.append(queue[offset])
                    count -= 1
            if count == 0:
                break
    return panel
