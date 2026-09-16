"""Corrected, role-independent state contract; v1 remains frozen for replay.

Participant slots are team then numeric participant ID, never lane/role labels.
The caller must supply the roster/team map available at match start. This does
not establish availability of the public API in a live match.
"""
from bisect import bisect_right
from dataclasses import dataclass
import re
from gameplay.state_value import StateBuilder as LegacyStateBuilder, State, DRAGONS

STATE_VERSION = 'objective_history_v2_participant_order'
ALIASES = {'INFERNAL': 'FIRE', 'CLOUD': 'AIR', 'OCEAN': 'WATER',
           'MOUNTAIN': 'EARTH'}

def soul_element(event):
    """Support observed Match-V5 name and prior schema aliases, fail conflicts."""
    known = set()
    for field in ('dragonSoul', 'soulType', 'name'):
        value = str(event.get(field) or '').strip().upper()
        if value.endswith('_DRAGON'):
            value = value[:-7]
        value = ALIASES.get(value, value)
        if value in DRAGONS and value != 'OTHER':
            known.add(value)
    if len(known) > 1:
        raise ValueError('conflicting dragon soul element fields')
    return next(iter(known)) if known else 'OTHER'

@dataclass
class StateV2(State):
    state_version: str
    unassigned_soul_events: int

class StateBuilder:
    state_version = STATE_VERSION

    def __init__(self, pack, node_names):
        tm = {int(k): int(v) for k, v in pack['meta']['team_map'].items()}
        if set(tm) != set(range(1, 11)) or any(list(tm.values()).count(t) != 5 for t in (100, 200)):
            raise ValueError('invalid participant roster/team map')
        self.participant_order = tuple(sorted(tm, key=lambda pid: (tm[pid], pid)))
        sanitized = dict(pack)
        sanitized['meta'] = dict(pack['meta'], role_slots={pid: slot for slot, pid in enumerate(self.participant_order)})
        events = []
        unassigned = []
        for source in pack['events']:
            if source.get('type') != 'DRAGON_SOUL_GIVEN':
                events.append(source)
                continue
            if int(source.get('teamId', 0) or 0) not in (100, 200):
                unassigned.append(int(source['timestamp']))
                continue  # No team ownership inferred from an unassigned event.
            event = dict(source)
            # Resolve only at query time: even malformed future data must not
            # change an earlier state's availability or contents.
            events.append(event)
        sanitized['events'] = events
        self._pack = sanitized
        self._node_names = node_names
        self._unassigned = sorted(unassigned)

    def at(self, query_ms):
        q = int(query_ms)
        events = []
        for source in self._pack['events']:
            if int(source.get('timestamp', -1)) > q:
                continue
            event = source
            if source.get('type') == 'DRAGON_SOUL_GIVEN':
                event = dict(source, dragonSoul=soul_element(source))
            events.append(event)
        builder = LegacyStateBuilder(dict(self._pack, events=events), self._node_names)
        state = builder.at(q)
        values = {re.sub(r'^slot(\d+)_', r'participant_slot\1_', key): value
                  for key, value in state.values.items()}
        return StateV2(values, state.query_ms, state.snapshot_ms, STATE_VERSION,
                       bisect_right(self._unassigned, q))

def state_matrix(states, names, expected_version=STATE_VERSION):
    """Version and exact feature-order guard for new training/prediction code."""
    import numpy as np
    if not all(isinstance(s, StateV2) and s.state_version == expected_version for s in states):
        raise ValueError('state version mismatch; regenerate states and refit model')
    if not all(list(s.values) == list(names) for s in states):
        raise ValueError('state feature schema mismatch')
    return np.asarray([[s.values[n] for n in names] for s in states], dtype=float)
