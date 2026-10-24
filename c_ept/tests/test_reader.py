import ast
import copy
import dataclasses
import errno
import inspect
import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pytest

from c_ept import config as C
from c_ept import reader
from c_ept.tests import synth
from c_ept.tests.conftest import read_json


def strip(events):
    """Events without ``realTimestamp`` (the wall-clock stamp of the collector, not part of the game record)."""
    return [{k: v for k, v in e.items() if k != 'realTimestamp'} for e in events]


def cache_events(key, patch):
    """``read_json`` (conftest) keeps the file name, which is the match key, out of the error of a missing file."""
    return read_json(C.CACHE_ROOT / patch / 'cache' / C.CACHE_SETS[patch] / f'{key}.events.json', 'cache events.json')


def cache_meta(key, patch):
    return read_json(C.CACHE_ROOT / patch / 'cache' / C.CACHE_SETS[patch] / f'{key}.meta.json', 'cache meta.json')


def raw_frames(key, patch):
    """Only ``info.frames`` of the raw timeline file (the rest of the file holds player identifiers)."""
    path = C.RAW_ROOT / patch / 'raw' / C.RAW_REGION[patch] / 'timeline' / f'{key}.json'
    return read_json(path, 'raw timeline file')['info']['frames']


ERROR_BODY = {'status': {'message': 'Data not found', 'status_code': 404}}      # what the API sends instead of a match
OTHER_PUUID = f'{synth.IDENTIFIER_SENTINELS[0]}-OF-ANOTHER-PLAYER'                # a puuid that no synthetic player has


def write_broken(tmp_path, case):
    """Write one small synthetic raw match with a single defect; return the arguments of ``read_match``.

    The synthetic match has 7 regular frames (0, 60,300, ..., 360,300 ms) and the game-end frame at T = 400,000 ms.
    """
    rec = synth.make_record(T_ms=400_000, key=f'SYN_{case}')
    timeline, detail = synth.raw_timeline(rec), synth.raw_detail(rec)
    frames, parts = timeline['info']['frames'], detail['info']['participants']
    if case == 'nine':                    # detail lists only 9 participants
        parts.pop()
    elif case == 'noend':                 # timeline without GAME_END
        frames[-1]['events'] = [e for e in frames[-1]['events'] if e['type'] != 'GAME_END']
    elif case == 'layout':                # participant 6 on team 100
        parts[5]['teamId'] = 100
    elif case == 'gone':                  # no detail file
        detail = None
    elif case == 'empty_participants':    # a list, but without participants
        detail['info']['participants'] = []
    elif case == 'empty_frames':          # a list, but without frames
        timeline['info']['frames'] = []
    elif case == 'detail_without_participants':          # an API error body saved as the detail file
        detail = ERROR_BODY
    elif case == 'participants_not_a_list':
        detail['info']['participants'] = 'oops'
    elif case == 'timeline_without_frames':              # an API error body saved as the timeline file
        timeline = ERROR_BODY
    elif case == 'frames_not_a_list':
        timeline['info']['frames'] = {'0': {}}
    elif case == 'champion_id_missing':
        del parts[2]['championId']
    elif case == 'champion_id_string':
        parts[2]['championId'] = '106'
    elif case == 'champion_id_float':
        parts[2]['championId'] = 106.5
    elif case == 'first_frame_not_at_zero':
        frames[0]['timestamp'] = 5
    elif case == 'frames_not_increasing':
        frames[2]['timestamp'] = frames[1]['timestamp']
    elif case == 'game_end_not_in_last_frame':           # GAME_END moved one frame back
        end = next(e for e in frames[-1]['events'] if e['type'] == 'GAME_END')
        frames[-1]['events'].remove(end)
        frames[-2]['events'].append(end)
    elif case == 'game_end_before_last_regular_frame':   # GAME_END earlier than the last regular frame (360,300 ms)
        next(e for e in frames[-1]['events'] if e['type'] == 'GAME_END')['timestamp'] = frames[-2]['timestamp'] - 1
    elif case == 'event_after_game_end':                 # an event 1 ms after GAME_END, in the same (last) frame
        frames[-1]['events'].append({'type': 'LEVEL_UP', 'timestamp': rec.T_ms + 1, 'participantId': 1, 'level': 18})
    elif case == 'event_in_the_same_ms_after_game_end':  # an event in the ms of GAME_END, listed after it
        frames[-1]['events'].append({'type': 'LEVEL_UP', 'timestamp': rec.T_ms, 'participantId': 1, 'level': 18})
    elif case == 'match_id_differs':                      # the detail file is the one of another match
        detail['metadata']['matchId'] = 'SYN_another_match'
    elif case == 'match_id_missing_in_the_timeline':      # a match id in one file only is a difference too
        del timeline['metadata']['matchId']
    elif case == 'match_id_missing_in_the_detail':
        del detail['metadata']['matchId']
    elif case == 'game_id_differs':                       # both files have a game id (the synthetic timeline has 0)
        detail['info']['gameId'] = 1_234_567
    elif case == 'puuid_differs_in_the_detail':           # participant 10 has another puuid in the detail than in the timeline
        parts[9]['puuid'] = OTHER_PUUID
    elif case.startswith('puuid_differs_'):               # participant N has another puuid in the timeline than in the detail
        timeline['info']['participants'][int(case.rsplit('_', 1)[1]) - 1]['puuid'] = OTHER_PUUID
    elif case == 'puuids_swapped':                        # participants 1 and 2 swap their puuids in the timeline: right set, wrong ids
        a, b = timeline['info']['participants'][:2]
        a['puuid'], b['puuid'] = b['puuid'], a['puuid']
    elif case == 'timeline_without_participants':         # no list to compare the detail with
        del timeline['info']['participants']
    elif case == 'timeline_participant_missing':          # participant 10 is not listed in the timeline
        timeline['info']['participants'].pop()
    elif case == 'timeline_participant_twice':            # participant 1 is listed twice, the second time with another puuid
        timeline['info']['participants'].append({'participantId': 1, 'puuid': OTHER_PUUID})
    elif case == 'frame_player_missing':                  # participant 10 has no entry in regular frame 1
        del frames[1]['participantFrames']['10']
    elif case == 'frame_players_null':                    # regular frame 2: participantFrames is null
        frames[2]['participantFrames'] = None
    elif case == 'frame_players_absent':                  # regular frame 1: no participantFrames at all
        del frames[1]['participantFrames']
    elif case == 'frame_player_entry_not_an_object':      # regular frame 3: the entry of participant 4 is a string
        frames[3]['participantFrames']['4'] = 'oops'
    elif case == 'frame_player_entry_empty':              # regular frame 2: the entry of participant 4 is an object without a value
        frames[2]['participantFrames']['4'] = {}
    elif case == 'frame_player_entry_id_only':            # regular frame 2: the entry of participant 4 holds its participantId only
        frames[2]['participantFrames']['4'] = {'participantId': 4}
    elif case == 'frame_player_entry_no_numbers':         # regular frame 2: sections and keys are there, but not one value is a number
        frames[2]['participantFrames']['4'] = {'participantId': 4, 'position': {}, 'level': None,
                                               'championStats': {'armor': None, 'health': 'n/a'}}
    elif case == 'first_frame_player_missing':            # the first frame is a regular frame like the others
        del frames[0]['participantFrames']['1']
    elif case == 'last_regular_frame_player_missing':     # frames[-1] is the game-end frame, frames[-2] the last regular one
        del frames[-2]['participantFrames']['5']
    else:
        raise ValueError(case)
    synth.write_raw(tmp_path, rec.patch, rec.key, timeline, detail)
    return rec.key, rec.patch, tmp_path


def test_reader_fields_and_order(raw_fixture):            # fixture: first FIT 15.18 cache key, read from the raw files
    r = reader.read_match(*raw_fixture)
    assert r.frame_num.shape[1:] == (10, 33) and r.frame_ts[0] == 0 and r.frame_ts[-1] < r.T_ms
    assert list(r.team) == [100] * 5 + [200] * 5
    ts = [e['timestamp'] for e in r.events]
    assert ts == sorted(ts) and r.events[-1]['type'] == 'GAME_END'
    same_key, same_patch = r.key == raw_fixture[0], r.patch == raw_fixture[1]        # booleans: a failure message must not print the key
    assert same_key and same_patch
    synth.assert_valid(r)


def test_record_has_no_identifier_strings(raw_fixture):
    s = json.dumps(dataclasses.asdict(reader.read_match(*raw_fixture)), default=str)
    found = [name for name in ('puuid', 'riotId', 'summonerName') if name in s]      # the names only, never the record text
    assert found == []


def test_reader_rejects_incomplete_match(tmp_path):        # detail with 9 participants, timeline without GAME_END, player 6 on team 100
    for case, code in (('nine', 'participants_not_10'), ('noend', 'no_game_end'), ('layout', 'pid_team_layout'), ('gone', 'missing_file')):
        with pytest.raises(reader.MatchRejected, match=code) as err:
            reader.read_match(*write_broken(tmp_path, case))
        assert err.value.code == code


def test_lists_that_are_present_but_unusable_stay_rejections(tmp_path):
    # real-data cases keep their codes: participants present but not 10, frames present but no GAME_END
    for case, code in (('empty_participants', 'participants_not_10'), ('empty_frames', 'no_game_end')):
        with pytest.raises(reader.MatchRejected, match=code):
            reader.read_match(*write_broken(tmp_path, case))


FORMAT_CASES = {               # case: part of the reason that read_match must give
    'detail_without_participants': 'info.participants',
    'participants_not_a_list': 'info.participants',
    'timeline_without_frames': 'info.frames',
    'frames_not_a_list': 'info.frames',
    'champion_id_missing': 'championId',
    'champion_id_string': 'championId',
    'champion_id_float': 'championId',
    'first_frame_not_at_zero': 'not at 0',
    'frames_not_increasing': 'do not increase',
    'game_end_not_in_last_frame': 'GAME_END is not in the last frame',
    'game_end_before_last_regular_frame': 'not before the game end',
    'event_after_game_end': 'after GAME_END',
    'event_in_the_same_ms_after_game_end': 'after GAME_END',
}


def test_structural_surprises_raise_match_format_error(tmp_path):
    """A file that is not a Match-V5 file as expected (an error body, a shifted clock, a missing champion id) is an error, not a
    rejection that would be counted like a short or incomplete match."""
    for case, part in FORMAT_CASES.items():
        key, patch, root = write_broken(tmp_path, case)
        with pytest.raises(reader.MatchFormatError) as err:
            reader.read_match(key, patch, root)
        e = err.value
        assert isinstance(e, ValueError) and not isinstance(e, reader.MatchRejected), case
        assert e.key == key and part in e.reason, case
        assert e.reason in str(e) and patch in str(e) and key not in str(e), case          # the message never holds the match key
        clone = pickle.loads(pickle.dumps(e))                                              # it must survive a worker pool
        assert (clone.key, clone.reason, str(clone)) == (e.key, e.reason, str(e)), case


def test_events_match_cache(raw_fixture):                  # raw events == cache events.json (same key)
    got, want = strip(reader.read_match(*raw_fixture).events), strip(cache_events(*raw_fixture))
    n_got, n_want = len(got), len(want)
    differ = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]       # event numbers only: an event holds the game id
    assert n_got == n_want and differ == []                                  # same length and no differing pair: the lists are equal


def test_runes_match_cache_meta(raw_fixture):              # runes_by_pid has the format of the cache meta static_meta.runes_by_pid
    r = reader.read_match(*raw_fixture)
    static = cache_meta(*raw_fixture)['static_meta']
    static = ast.literal_eval(static) if isinstance(static, str) else static
    assert r.runes_by_pid == static['runes_by_pid']
    assert list(r.champ) == [static['champion_by_pid'][str(p)] for p in range(1, 11)]


def test_frame_values_and_end_xy_match_raw_file(raw_fixture):
    """Spelled out with the raw field names, independently of config.FRAME_FIELDS: column order, ``cs_`` prefix, end frame."""
    r = reader.read_match(*raw_fixture)
    frames = raw_frames(*raw_fixture)
    assert len(r.frame_ts) == len(frames) - 1              # the last frame (game end) is not a regular frame
    for k in (0, 2, len(r.frame_ts) - 1):
        assert r.frame_ts[k] == frames[k]['timestamp']
        for pid in range(1, 11):
            pf = frames[k]['participantFrames'][str(pid)]
            want = [pf['position']['x'], pf['position']['y'], pf['level'], pf['xp'], pf['currentGold'], pf['totalGold'],
                    pf['minionsKilled'], pf['jungleMinionsKilled']]
            want += [pf['championStats'][name] for name in (
                'abilityHaste', 'abilityPower', 'armor', 'armorPen', 'armorPenPercent', 'attackDamage', 'attackSpeed',
                'bonusArmorPenPercent', 'bonusMagicPenPercent', 'ccReduction', 'cooldownReduction', 'health', 'healthMax',
                'healthRegen', 'lifesteal', 'magicPen', 'magicPenPercent', 'magicResist', 'movementSpeed', 'omnivamp',
                'physicalVamp', 'power', 'powerMax', 'powerRegen', 'spellVamp')]
            assert list(r.frame_num[k, pid - 1]) == want
    for pid in range(1, 11):
        pos = frames[-1]['participantFrames'][str(pid)]['position']
        assert list(r.end_xy[pid - 1]) == [pos['x'], pos['y']]


def test_record_repr_hides_the_match_key(raw_fixture):
    """A record shows up in assertion explanations ("where x = MatchRecord(...).patch"), tracebacks and logs: its repr names the patch,
    T_ms and the shape of the frames, and not the match key. Booleans only below: a failure message must not print the key either."""
    for rec in (synth.make_record(key='SYN_repr_key_17', patch='15.19'), reader.read_match(*raw_fixture)):
        text = ' | '.join((repr(rec), str(rec), repr([rec]), repr({'record': rec}), f'{rec!r}', format(rec)))
        key_hidden = rec.key not in text
        names_the_patch = f'patch={rec.patch!r}' in text and f'T_ms={rec.T_ms}' in text
        marks_the_redaction = 'key=<redacted>' in text and f'frame_num={tuple(rec.frame_num.shape)}' in text
        assert key_hidden and names_the_patch and marks_the_redaction


def test_synthetic_roundtrip_through_raw_files(tmp_path):
    """record -> raw JSON -> record: missing frame fields are NaN, identifiers are never read, events keep their order. (A player who is
    missing from a whole regular frame is not a NaN row any more: that match is rejected, see ``CROSS_CASES``.)"""
    rec = synth.make_record(
        T_ms=310_000, patch='15.19', key='SYN_roundtrip',
        kills=[dict(t=200_000, x=1_000, y=2_000, killer=1, victim=6, assists=(2, 3), bounty=250, shutdown=100),
               dict(t=200_000, x=1_100, y=2_000, killer=7, victim=2),                    # same ms: keeps its order
               dict(t=300_000, x=None, y=None, killer=0, victim=8)],                     # execution without position
        objectives=[dict(t=250_000, type='ELITE_MONSTER_KILL', x=9_800, y=4_400, killer=7, assists=(8,), team=200,
                         monster='DRAGON', sub='EARTH_DRAGON'),
                    dict(t=280_000, type='BUILDING_KILL', x=4_300, y=13_800, team=200, building='TOWER_BUILDING',
                         tower='OUTER_TURRET', lane='TOP_LANE')])
    num = rec.frame_num.copy()
    num[2, 3, :2] = np.nan                  # no position of player 4 in frame 2
    num[3, 4, 10] = np.nan                  # one championStats value of player 5 in frame 3
    end = rec.end_xy.copy()
    end[0] = np.nan
    rec = dataclasses.replace(rec, frame_num=num, end_xy=end)
    synth.write_raw(tmp_path, rec.patch, rec.key, synth.raw_timeline(rec), synth.raw_detail(rec))

    back = reader.read_match(rec.key, rec.patch, tmp_path)
    assert back == rec
    assert np.isnan(back.frame_num[2, 3, :2]).all() and np.isnan(back.frame_num[3, 4, 10]) and np.isnan(back.end_xy[0]).all()
    assert [e['type'] for e in back.events if e['timestamp'] == 200_000] == ['CHAMPION_KILL', 'CHAMPION_KILL']
    assert [e['killerId'] for e in back.events if e['timestamp'] == 200_000] == [1, 7]
    s = json.dumps(dataclasses.asdict(back), default=str)
    assert not any(sentinel in s for sentinel in synth.IDENTIFIER_SENTINELS)


def test_synth_records_are_valid_deterministic_and_picklable():
    a = synth.records_from_seed('SYN_17', '15.19')
    assert a == synth.records_from_seed('SYN_17', '15.19') and a != synth.records_from_seed('SYN_18', '15.19')
    assert (a.key, a.patch) == ('SYN_17', '15.19') and pickle.loads(pickle.dumps(a)) == a
    assert pickle.loads(pickle.dumps(synth.records_from_seed)) is synth.records_from_seed        # a module-level function: workers get it
    kills = [e for e in a.events if e['type'] == 'CHAMPION_KILL']
    assert len(kills) > 10 and any(e['type'] == 'ELITE_MONSTER_KILL' for e in a.events)
    assert all(e['killerId'] == 0 or (e['killerId'] <= 5) != (e['victimId'] <= 5) for e in kills)        # kills cross the teams
    for rec in (a, synth.make_record(), synth.make_record(T_ms=425_000), synth.make_record(T_ms=425_000, n_frames=3)):
        synth.assert_valid(rec)
    r = synth.make_record(T_ms=425_000)                                    # frames 0, 60,300, ..., 420,300
    assert list(r.frame_ts) == [0] + [60_000 * k + 300 for k in range(1, 8)]
    with pytest.raises(ValueError):
        synth.make_record(T_ms=100_000, kills=[dict(t=100_001, x=0, y=0, killer=1, victim=6)])
    with pytest.raises(ValueError):
        synth.make_record(T_ms=125_000, n_frames=4)                        # frame 3 would lie after the game end


def test_record_arrays_are_read_only(raw_fixture):
    """Every numpy array of a MatchRecord is read-only, however the record came about (read, built, replaced, pickled, copied).
    The events (plain dicts in a tuple) and runes_by_pid are not frozen; they are only not to be modified."""
    base = synth.make_record()
    records = [reader.read_match(*raw_fixture), base, synth.records_from_seed('SYN_5', '15.19'),
               dataclasses.replace(base, frame_ts=np.array([0, 60_300], dtype=np.int64), frame_num=base.frame_num[:2].copy())]
    records += [pickle.loads(pickle.dumps(records[0])), copy.deepcopy(records[1])]
    for rec in records:
        arrays = {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec) if isinstance(getattr(rec, f.name), np.ndarray)}
        assert set(arrays) == {'team', 'champ', 'frame_ts', 'frame_num', 'end_xy'}
        for name, a in arrays.items():
            assert not a.flags.writeable, name
            with pytest.raises(ValueError):
                a[(0,) * a.ndim] = 0
        assert type(rec.events) is tuple and all(type(e) is dict for e in rec.events) and type(rec.runes_by_pid) is dict


# ---- pre-S2 guard 1: the holdout gate ---------------------------------------------------------------------------------------------
# 15.22 (TEST) and 16.13 (EXTERNAL) are opened once, at S8. None of these tests opens a raw file of either patch: they use a root folder
# that does not exist, or synthetic files under the tmp_path of pytest.
HOLDOUT = ('15.22', '16.13')


class _RootThatFailsWhenUsed(os.PathLike):
    """A ``root`` that raises the moment anything turns it into a path: ``read_match`` does that when it first goes for a file."""

    def __fspath__(self):
        raise AssertionError('read_match used the root: it went for a file before the gate')


@pytest.mark.parametrize('patch', HOLDOUT)
def test_holdout_patches_are_refused_before_any_file_access(tmp_path, monkeypatch, patch):
    root = tmp_path / 'no_such_root'                                       # a folder that does not exist
    with pytest.raises(reader.MatchRejected, match='missing_file'):        # an ordinary patch goes on to look for its files, finds none ...
        reader.read_match('SYN_gate', '15.18', root)
    with pytest.raises(reader.HoldoutRefused):                             # ... a holdout patch is stopped before it looks
        reader.read_match('SYN_gate', patch, root)
    assert not root.exists()
    reads = []
    monkeypatch.setattr(reader, '_read_json', lambda path, *args: reads.append(path))
    with pytest.raises(reader.HoldoutRefused):                             # not even the root is turned into a path
        reader.read_match('SYN_gate', patch, _RootThatFailsWhenUsed())
    with pytest.raises(reader.HoldoutRefused):                             # the switch spelled out as False
        reader.read_match(key='SYN_gate', patch=patch, root=root, allow_holdout=False)
    assert reads == []


def test_the_refusal_is_a_value_error_that_names_the_patch_and_the_rule_and_never_the_key(tmp_path):
    key = 'SYN_secret_key_77'
    messages = {}
    for patch in HOLDOUT:
        with pytest.raises(reader.HoldoutRefused) as err:
            reader.read_match(key, patch, tmp_path / 'no_such_root')
        e, text = err.value, str(err.value)
        messages[patch] = text
        assert isinstance(e, ValueError) and not isinstance(e, (reader.MatchRejected, reader.MatchFormatError))
        assert patch in text and 'opened once at S8' in text and 'allow_holdout' in text
        assert key not in text and key not in repr(e) and 'secret' not in text
        clone = pickle.loads(pickle.dumps(e))                              # it must survive a worker pool
        assert type(clone) is reader.HoldoutRefused and str(clone) == text and clone.patch == patch
    assert 'TEST' in messages['15.22'] and 'EXTERNAL' in messages['16.13']      # the message says which role the patch holds


def test_the_gate_is_a_keyword_only_switch_that_is_off_by_default(tmp_path):
    p = inspect.signature(reader.read_match).parameters['allow_holdout']
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    with pytest.raises(TypeError):
        reader.read_match('SYN_gate', '15.18', tmp_path, True)             # a fourth positional argument does not reach the switch


def test_allow_holdout_opens_the_gate_and_nothing_else(tmp_path):
    for patch in HOLDOUT:                                                  # past the gate, on a root without files: the ordinary rejection
        with pytest.raises(reader.MatchRejected, match='missing_file'):
            reader.read_match('SYN_gate', patch, tmp_path / 'no_such_root', allow_holdout=True)
    rec = synth.make_record(T_ms=400_000, key='SYN_gate', patch='15.22')   # synthetic files under tmp_path, not the raw data
    synth.write_raw(tmp_path, rec.patch, rec.key, synth.raw_timeline(rec), synth.raw_detail(rec))
    assert reader.read_match(rec.key, rec.patch, tmp_path, allow_holdout=True) == rec
    with pytest.raises(reader.HoldoutRefused):                             # the same files without the switch
        reader.read_match(rec.key, rec.patch, tmp_path)
    ordinary = synth.make_record(T_ms=400_000, key='SYN_gate', patch='15.19')
    synth.write_raw(tmp_path, ordinary.patch, ordinary.key, synth.raw_timeline(ordinary), synth.raw_detail(ordinary))
    read = reader.read_match(ordinary.key, ordinary.patch, tmp_path)
    assert read == ordinary and read == reader.read_match(ordinary.key, ordinary.patch, tmp_path, allow_holdout=True)


@pytest.mark.parametrize('patch', HOLDOUT)
@pytest.mark.parametrize('not_true', [1, 1.0, 'yes', [True]], ids=['int_1', 'float_1', 'str_yes', 'list'])
def test_only_the_constant_true_opens_the_gate(tmp_path, patch, not_true):
    """The switch opens the gate as the constant ``True`` (``is True``), not by truthiness: a flag that happens to be 1, 1.0, "yes" or
    a non-empty list is refused. Synthetic files under tmp_path: with the flag accepted the match would be read, so a refusal is the gate."""
    rec = synth.make_record(T_ms=400_000, key='SYN_gate', patch=patch)
    synth.write_raw(tmp_path, rec.patch, rec.key, synth.raw_timeline(rec), synth.raw_detail(rec))
    with pytest.raises(reader.HoldoutRefused):
        reader.read_match(rec.key, rec.patch, tmp_path, allow_holdout=not_true)
    assert reader.read_match(rec.key, rec.patch, tmp_path, allow_holdout=True) == rec           # the constant True still opens it


def test_the_holdout_patches_are_the_test_and_external_patches_of_the_config():
    assert set(reader.HOLDOUT_PATCHES) == {'15.22', '16.13'} == set(HOLDOUT)
    assert set(reader.HOLDOUT_PATCHES) <= set(C.PATCHES) and not set(reader.HOLDOUT_PATCHES) & set(C.PATCHES[:4])
    assert {reader.HOLDOUT_ROLES[p] for p in reader.HOLDOUT_PATCHES} == {'TEST', 'EXTERNAL'} <= set(C.ROLES)


def test_the_holdout_table_is_every_patch_beyond_the_four_s1_patches_of_the_config():
    """The gate looks a patch up in ``reader.HOLDOUT_ROLES``, a table written out in the reader. This ties it to the config: every patch
    after the four that S1 reads is a holdout and no other is. A patch added to ``config.PATCHES`` without a line in the table (a hole in
    the gate) fails here, and so does a line for a patch that S1 reads. The table stays written out: the config lists the patches and the
    roles but not which patch holds which role, and the roles go into the refusal message. ``reader.HOLDOUT_PATCHES`` is built from the
    table (``tuple(HOLDOUT_ROLES)``), so comparing the two would test nothing; it is compared with the config, in the config's order."""
    beyond = tuple(p for p in C.PATCHES if p not in C.PATCHES[:4])               # what the config says: the patches after the four of S1
    assert beyond and set(beyond) == set(reader.HOLDOUT_ROLES)
    assert reader.HOLDOUT_PATCHES == beyond


def package_modules(package_dir: Path) -> list[str]:
    """Every ``.py`` file under ``package_dir`` (paths relative to it, with ``/``): the modules of the package, subfolders included
    (``checks/``), found when this runs, so there is no list to keep up to date. Everything under ``tests/`` is left out: the tests are
    not part of the package."""
    found = (p.relative_to(package_dir).as_posix() for p in package_dir.rglob('*.py'))
    return sorted(rel for rel in found if rel.split('/')[0] != 'tests')


def is_constant_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def gate_openers(source: str) -> list[int]:
    """Line numbers of the places in ``source`` that pass ``allow_holdout`` anything but the constant False (True, a variable, ...): a
    call with that keyword (``read_match(..., allow_holdout=True)``; ``dict(allow_holdout=True)`` and ``partial(f, allow_holdout=flag)``
    are calls too) or a dict literal with that key (``read_match(..., **{'allow_holdout': True})``, or ``opts = {'allow_holdout': flag}``
    for a later ``**opts``). A bare ``**kwargs`` is not judged: it can be a wrapper that passes on whatever it was given."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            lines += [node.lineno for kw in node.keywords if kw.arg == 'allow_holdout' and not is_constant_false(kw.value)]
        elif isinstance(node, ast.Dict):
            lines += [node.lineno for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and key.value == 'allow_holdout' and not is_constant_false(value)]
    return sorted(lines)


def gate_offenders(package_dir: Path) -> list[str]:
    """``<module>:<line>`` of every place in the modules of the package that opens the gate."""
    return [f'{rel}:{line}' for rel in package_modules(package_dir)
            for line in gate_openers((package_dir / rel).read_text(encoding='utf-8'))]


def test_the_scan_for_calls_that_open_the_gate_sees_them():
    assert gate_openers('read_match(k, p, allow_holdout=True)') == [1]
    assert gate_openers('x = 1\nreader.read_match(k, p, root, allow_holdout=flag)') == [2]
    assert gate_openers('f(allow_holdout=1)') == [1]
    assert gate_openers('read_match(k, p, allow_holdout=False)\nread_match(k, p)\nf(allowed=True)') == []


def test_the_scan_for_calls_that_open_the_gate_sees_a_dict_of_keywords_too():
    assert gate_openers("read_match(k, p, **{'allow_holdout': True})") == [1]                       # the keyword dict written in the call
    assert gate_openers('read_match(k, p,\n           **{"allow_holdout": flag})') == [2]           # a variable is not the constant False
    assert gate_openers("opts = {'allow_holdout': True}\nread_match(k, p, **opts)") == [1]          # the dict is built elsewhere
    assert gate_openers("kw = dict(allow_holdout=True)\nread_match(k, p, **kw)") == [1]             # dict(...) is a call with the keyword
    assert gate_openers('functools.partial(read_match, allow_holdout=True)') == [1]
    assert gate_openers("read_match(k, p, **{'allow_holdout': False})\nf(**{'other': True})\nf(**kwargs)\ng({**base, 'x': 1})") == []


def test_the_scan_of_the_package_for_the_gate_goes_into_the_subfolders_and_leaves_out_the_tests(tmp_path):
    opener = "read_match(k, p, **{'allow_holdout': True})"
    files = {'clean.py': 'read_match(k, p, allow_holdout=False)', 'top.py': 'x = 1\n' + opener,
             'tests/__init__.py': '', 'tests/test_x.py': opener,                                   # the tests are not part of the package
             'checks/__init__.py': '', 'checks/probe.py': opener, 'a/b/deeper.py': 'f(allow_holdout=True)'}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    assert package_modules(tmp_path) == sorted(name for name in files if not name.startswith('tests/'))
    assert sorted(gate_offenders(tmp_path)) == ['a/b/deeper.py:1', 'checks/probe.py:1', 'top.py:2']
    (tmp_path / 'later.py').write_text('f(allow_holdout=flag)', encoding='utf-8')
    assert 'later.py:1' in gate_offenders(tmp_path)                                         # found, with no list to keep up to date


def test_nothing_in_c_ept_opens_the_gate():
    """The switch stays False everywhere in the package until S8 (which has its own approval): no call in a c_ept module, in any
    subfolder (``checks/`` included), passes it anything but the constant False. The tests are not part of the package."""
    package = Path(reader.__file__).parent
    scanned = package_modules(package)
    assert {'reader.py', 'run_s1.py', 'traits.py', 'checks/common.py', 'checks/sweep_s1.py'} <= set(scanned)       # the scan is not empty
    assert not any(rel.startswith('tests/') for rel in scanned)
    assert gate_offenders(package) == []


# The gate is the first thing read_match does, and it protects nothing if another module builds a raw path itself from the root and the
# region of the config. So no module of the package names them, except the reader (which reads behind its gate), the config (which
# defines them) and the tests (which write and read synthetic files). A guard against a slip, not a wall: a string such as
# ``getattr(C, 'RAW_ROOT')`` is not seen. The string constants exactly equal to ``'raw'`` (the folder below a patch) and
# ``'C_EPT_RAW_ROOT'`` (the environment variable of the raw root) are, so that the realistic slip ``C.CACHE_ROOT / patch / 'raw' / 'kr'``
# is caught; they are compared for equality, so a docstring that speaks of raw data is not.
RAW_LOCATION_NAMES = ('RAW_ROOT', 'RAW_REGION')
RAW_LOCATION_STRINGS = ('raw', 'C_EPT_RAW_ROOT')
RAW_LOCATION_EXEMPT = ('reader.py', 'config.py')               # paths relative to the package folder; everything under tests/ too


def raw_location_users(source: str) -> list[int]:
    """Line numbers of the places in ``source`` that name the raw-file location: as a name (``RAW_ROOT``), as an attribute
    (``C.RAW_ROOT``), as an imported name (``from c_ept.config import RAW_ROOT as ROOT``: the import is the one place where it shows) or
    as a string constant exactly equal to ``'raw'`` or ``'C_EPT_RAW_ROOT'``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in RAW_LOCATION_STRINGS:
            lines.append(node.lineno)
            continue
        else:
            continue
        if name in RAW_LOCATION_NAMES:
            lines.append(node.lineno)
    return sorted(lines)


def modules_to_scan(package_dir: Path) -> list[str]:
    """The ``.py`` files under ``package_dir`` (paths relative to it, with ``/``) that must not name the raw-file location: every one
    that is there when this runs, subfolders included, except the exempt modules and everything under ``tests/``."""
    return [rel for rel in package_modules(package_dir) if rel not in RAW_LOCATION_EXEMPT]


def raw_location_offenders(package_dir: Path) -> list[str]:
    """``<module>:<line>`` of every place that names the raw-file location in the modules to scan."""
    return [f'{rel}:{line}' for rel in modules_to_scan(package_dir)
            for line in raw_location_users((package_dir / rel).read_text(encoding='utf-8'))]


def test_the_scan_for_the_raw_file_location_sees_a_name_an_attribute_and_an_import():
    assert raw_location_users('base = C.RAW_ROOT / patch') == [1]                       # an attribute of the config module
    assert raw_location_users('x = 1\nregion = C.RAW_REGION[patch]') == [2]
    assert raw_location_users('RAW_ROOT / patch') == [1]                                # a bare name
    assert raw_location_users('import c_ept.config as cfg\nroot = cfg.RAW_ROOT') == [2]
    assert raw_location_users('from c_ept import config\nconfig.RAW_ROOT, config.RAW_REGION') == [2, 2]
    assert raw_location_users('from c_ept.config import RAW_ROOT as ROOT') == [1]       # an import under another name
    assert raw_location_users('"""reads RAW_ROOT"""\n# RAW_REGION\nx = C.CACHE_ROOT\ny = RAW_ROOTS + C_EPT_RAW_ROOT') == []   # not names


def test_the_scans_see_a_relative_import_as_they_see_an_absolute_one():
    """Neither scan of this file reads the dotted name of a module: they key on keyword names, attribute names and import aliases. So
    ``from .config import RAW_ROOT`` and ``from . import config`` need no resolving against the package, as the import scans of
    test_tokens.py and test_terrain.py do (``conftest.imported_names``; final review, Task 12 deferred). Pinned here for the spellings
    a module of ``c_ept/`` would use."""
    assert raw_location_users('from .config import RAW_ROOT') == [1]
    assert raw_location_users('from .config import RAW_REGION as REGION') == [1]
    assert raw_location_users('from . import config\nbase = config.RAW_ROOT / patch') == [2]
    assert raw_location_users('from .. import config as cfg\nx = 1\nregion = cfg.RAW_REGION[patch]') == [3]
    assert raw_location_users('from .config import *\nbase = RAW_ROOT') == [2]                       # a star import: the name is used
    assert raw_location_users('from . import config_values\nfrom .config import RAW_ROOTS') == []     # other names are not the names
    assert gate_openers('from .reader import read_match as rm\nrm(k, p, allow_holdout=True)') == [2]       # any call with the keyword


def test_the_scan_for_the_raw_file_location_sees_the_strings_of_the_path_too():
    """The realistic slip builds the path from the cache root, which is a different name: ``C.CACHE_ROOT / patch / 'raw' / 'kr'``. The
    string constants exactly equal to ``'raw'`` and ``'C_EPT_RAW_ROOT'`` are flagged, however they are quoted."""
    assert raw_location_users("base = C.CACHE_ROOT / patch / 'raw' / 'kr'") == [1]
    assert raw_location_users('x = 1\np = Path(root) / patch / "raw" / region') == [2]
    assert raw_location_users("import os\nroot = os.environ.get('C_EPT_RAW_ROOT')") == [2]
    assert raw_location_users("os.environ['C_EPT_RAW_ROOT']") == [1]


def test_the_scan_for_the_raw_file_location_compares_strings_for_equality_not_by_substring():
    docs = '"""reads the raw data of RAW_ROOT"""\nx = "a raw file"\ny = "raw/timeline"\nz = f"{k}/raw"\nw = "RAWS"\nv = "C_EPT_RAW_ROOT_X"'
    assert raw_location_users(docs) == []
    assert raw_location_users('"raw"\n"C_EPT_RAW_ROOT"') == [1, 2]                         # the exact strings, even as bare statements


def test_the_scan_of_the_package_lists_the_folder_when_it_runs_and_skips_only_the_exempt_modules(tmp_path):
    files = {'reader.py': 'base = C.RAW_ROOT', 'config.py': 'RAW_ROOT = 1',                    # exempt
             'tests/__init__.py': '', 'tests/test_x.py': 'C.RAW_REGION',                       # the tests are exempt
             'clean.py': 'root = C.CACHE_ROOT  # not RAW_ROOT',
             'later.py': 'base = C.RAW_ROOT / patch',                                          # a module that is added later
             'slip.py': "base = C.CACHE_ROOT / patch / 'raw' / 'kr'",                          # the cache root with the raw folder
             'sub/__init__.py': '', 'sub/deeper.py': '\nregion = C.RAW_REGION[patch]',         # a subpackage is scanned
             'sub/reader.py': 'RAW_ROOT'}                                                      # only the reader of the package is exempt
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    assert sorted(raw_location_offenders(tmp_path)) == ['later.py:1', 'slip.py:1', 'sub/deeper.py:2', 'sub/reader.py:1']
    (tmp_path / 'newer.py').write_text('import c_ept.config as C\nC.RAW_ROOT', encoding='utf-8')
    assert 'newer.py:2' in raw_location_offenders(tmp_path)                                    # found, with no list to keep up to date


def test_no_module_of_c_ept_but_the_reader_and_the_config_names_the_raw_file_location():
    package = Path(reader.__file__).parent
    scanned = modules_to_scan(package)
    assert 'run_s1.py' in scanned and 'reader.py' not in scanned and 'config.py' not in scanned       # the scan is not empty
    assert {'checks/common.py', 'checks/sweep_s1.py'} <= set(scanned)                       # the subfolder of the scripts is in it
    assert not any(rel.startswith('tests/') for rel in scanned)
    assert raw_location_offenders(package) == []


# ---- pre-S2 guard 2: the timeline and the detail file must belong together, and every regular frame must hold all ten players --------
# Rejections, not format errors: this is real data that contradicts itself (the final review's probe found none among the 28,701 S1
# matches). The puuids of the synthetic files are sentinels; the checks compare them in memory and nothing may show them.
CROSS_CASES = {              # case (see write_broken): the code that read_match must reject it with
    'match_id_differs': 'timeline_detail_mismatch',
    'match_id_missing_in_the_timeline': 'timeline_detail_mismatch',
    'match_id_missing_in_the_detail': 'timeline_detail_mismatch',
    'game_id_differs': 'timeline_detail_mismatch',
    **{f'puuid_differs_{pid}': 'timeline_detail_mismatch' for pid in range(1, 11)},
    'puuid_differs_in_the_detail': 'timeline_detail_mismatch',
    'puuids_swapped': 'timeline_detail_mismatch',
    'timeline_without_participants': 'timeline_detail_mismatch',
    'timeline_participant_missing': 'timeline_detail_mismatch',
    'timeline_participant_twice': 'timeline_detail_mismatch',
    'frame_player_missing': 'incomplete_frame',
    'frame_players_null': 'incomplete_frame',
    'frame_players_absent': 'incomplete_frame',
    'frame_player_entry_not_an_object': 'incomplete_frame',
    'frame_player_entry_empty': 'incomplete_frame',                  # an entry that gives no value is a missing player, not a NaN row
    'frame_player_entry_id_only': 'incomplete_frame',
    'frame_player_entry_no_numbers': 'incomplete_frame',
    'first_frame_player_missing': 'incomplete_frame',
    'last_regular_frame_player_missing': 'incomplete_frame',
}


def write_variant(tmp_path, key, *edits, patch='15.18'):
    """Write one synthetic raw match (T = 400,000 ms) after ``edit(timeline, detail)`` has changed the two file dicts for every edit in
    ``edits``; returns the record the unchanged files stand for."""
    rec = synth.make_record(T_ms=400_000, key=key, patch=patch)
    timeline, detail = synth.raw_timeline(rec), synth.raw_detail(rec)
    for edit in edits:
        edit(timeline, detail)
    synth.write_raw(tmp_path, rec.patch, rec.key, timeline, detail)
    return rec


def other_match_id(timeline, detail):
    detail['metadata']['matchId'] = 'SYN_another_match'


def player_missing_in_a_frame(timeline, detail):
    del timeline['info']['frames'][1]['participantFrames']['10']


def player_with_an_empty_entry_in_a_frame(timeline, detail):
    timeline['info']['frames'][1]['participantFrames']['10'] = {}                       # the entry is there and holds no value


def no_game_end(timeline, detail):
    last = timeline['info']['frames'][-1]
    last['events'] = [e for e in last['events'] if e['type'] != 'GAME_END']


def first_frame_late(timeline, detail):
    timeline['info']['frames'][0]['timestamp'] = 5


@pytest.mark.parametrize('case, code', list(CROSS_CASES.items()))
def test_a_pair_that_contradicts_itself_is_a_counted_rejection(tmp_path, case, code):
    key, patch, root = write_broken(tmp_path, case)
    with pytest.raises(reader.MatchRejected) as err:
        reader.read_match(key, patch, root)
    e = err.value
    assert e.code == code and str(e) == code and not isinstance(e, ValueError)         # a rejection, not a MatchFormatError
    text = f'{e} {e!r} {e.args}'
    assert key not in text and 'another' not in text.lower() and not any(s in text for s in synth.IDENTIFIER_SENTINELS)
    clone = pickle.loads(pickle.dumps(e))
    assert type(clone) is reader.MatchRejected and clone.code == code


def test_pairs_that_agree_are_read_whatever_the_order_or_the_optional_ids(tmp_path):
    """Compared by participant id, not by position; a game id is compared only when both files have one; the game-end frame is not a
    regular frame, so a player missing there only leaves that player's end position NaN."""
    def game_id_in_both(timeline, detail):
        timeline['info']['gameId'] = detail['info']['gameId'] = 987_654_321

    def game_id_only_in_the_detail(timeline, detail):
        del timeline['info']['gameId']
        detail['info']['gameId'] = 987_654_321

    def game_id_null_in_the_timeline(timeline, detail):
        timeline['info']['gameId'] = None
        detail['info']['gameId'] = 987_654_321

    def timeline_participants_reversed(timeline, detail):
        timeline['info']['participants'].reverse()

    def detail_participants_reversed(timeline, detail):
        detail['info']['participants'].reverse()

    for name, edit in (('both', game_id_in_both), ('detail', game_id_only_in_the_detail), ('null', game_id_null_in_the_timeline),
                       ('tl_reversed', timeline_participants_reversed), ('dt_reversed', detail_participants_reversed)):
        sub = tmp_path / name
        rec = write_variant(sub, 'SYN_agree', edit)
        assert reader.read_match(rec.key, rec.patch, sub) == rec, name
    sub = tmp_path / 'end'

    def player_missing_in_the_game_end_frame(timeline, detail):
        del timeline['info']['frames'][-1]['participantFrames']['1']
    rec = write_variant(sub, 'SYN_agree', player_missing_in_the_game_end_frame)
    back = reader.read_match(rec.key, rec.patch, sub)
    assert np.isnan(back.end_xy[0]).all() and not np.isnan(back.end_xy[1:]).any() and np.array_equal(back.end_xy[1:], rec.end_xy[1:])
    assert np.array_equal(back.frame_num, rec.frame_num) and back.events == rec.events


def test_an_entry_without_a_value_in_the_game_end_frame_is_read_like_a_missing_player(tmp_path):
    """The game-end frame is not a regular frame: an entry that holds no value there is accepted and leaves only that player's end
    position NaN, as a missing entry does. (A regular frame rejects the same entries, see ``CROSS_CASES``.)"""
    for name, entry in (('empty', {}), ('id_only', {'participantId': 1})):
        def edit(timeline, detail, entry=entry):
            timeline['info']['frames'][-1]['participantFrames']['1'] = dict(entry)
        sub = tmp_path / name
        rec = write_variant(sub, 'SYN_agree', edit)
        back = reader.read_match(rec.key, rec.patch, sub)
        assert np.isnan(back.end_xy[0]).all() and np.array_equal(back.end_xy[1:], rec.end_xy[1:]), name
        assert np.array_equal(back.frame_num, rec.frame_num) and back.events == rec.events, name


def test_an_entry_with_one_usable_value_is_read_with_the_rest_missing(tmp_path):
    """The rule for a regular frame is "no usable value at all", not "few values": an entry with a position only, with one stat only,
    or with everything but the position (a legal missing position, as in the S1 store) is read, the missing values are NaN, and the
    match is accepted."""
    def entry_of_player_4(entry):
        def edit(timeline, detail):
            timeline['info']['frames'][2]['participantFrames']['4'] = entry
        return edit

    def no_position(timeline, detail):
        del timeline['info']['frames'][2]['participantFrames']['4']['position']

    cases = {'position_only': (entry_of_player_4({'participantId': 4, 'position': {'x': 1_234, 'y': 5_678}}), {'x': 1_234.0, 'y': 5_678.0}),
             'one_stat_only': (entry_of_player_4({'participantId': 4, 'level': 7}), {'level': 7.0})}
    for name, (edit, want) in cases.items():
        sub = tmp_path / name
        rec = write_variant(sub, 'SYN_partial', edit)
        row = reader.read_match(rec.key, rec.patch, sub).frame_num[2, 3]
        got = {C.FRAME_FIELDS[j]: float(row[j]) for j in np.flatnonzero(~np.isnan(row))}
        assert got == want, name
    sub = tmp_path / 'no_position'
    rec = write_variant(sub, 'SYN_partial', no_position)
    row = reader.read_match(rec.key, rec.patch, sub).frame_num[2, 3]
    assert np.isnan(row[:2]).all() and np.array_equal(row[2:], rec.frame_num[2, 3, 2:])


@pytest.mark.parametrize('incomplete', [player_missing_in_a_frame, player_with_an_empty_entry_in_a_frame],
                         ids=['entry_deleted', 'entry_empty'])
def test_the_new_checks_only_reject_what_was_read_before(tmp_path, incomplete):
    """They run after every older check: a pair that fails an older check is rejected (or raises) as before, even when it fails a new
    one as well, and the mismatch comes before the incomplete frame. So no old count can change; only an accepted match can be lost.
    The incomplete frame is a player whose entry is deleted or whose entry is empty (one code path in ``_fill_frame``, both pinned: the
    order of the checks is the same for both)."""
    rec = write_variant(tmp_path / 'alone', 'SYN_order', incomplete)             # on its own it is an incomplete frame ...
    with pytest.raises(reader.MatchRejected) as err:
        reader.read_match(rec.key, rec.patch, tmp_path / 'alone')
    assert err.value.code == 'incomplete_frame'
    both_new = (other_match_id, incomplete)                                      # ... and next to the mismatch it comes second
    rec = write_variant(tmp_path / 'a', 'SYN_order', *both_new)
    with pytest.raises(reader.MatchRejected) as err:
        reader.read_match(rec.key, rec.patch, tmp_path / 'a')
    assert err.value.code == 'timeline_detail_mismatch'
    rec = write_variant(tmp_path / 'b', 'SYN_order', no_game_end, *both_new)
    with pytest.raises(reader.MatchRejected) as err:
        reader.read_match(rec.key, rec.patch, tmp_path / 'b')
    assert err.value.code == 'no_game_end'
    rec = write_variant(tmp_path / 'c', 'SYN_order', first_frame_late, *both_new)
    with pytest.raises(reader.MatchFormatError, match='not at 0'):
        reader.read_match(rec.key, rec.patch, tmp_path / 'c')
    rec = write_variant(tmp_path / 'd', 'SYN_order', lambda timeline, detail: detail['info']['participants'].pop(), *both_new)
    with pytest.raises(reader.MatchRejected) as err:
        reader.read_match(rec.key, rec.patch, tmp_path / 'd')
    assert err.value.code == 'participants_not_10'


# ---- key-free errors for a match file that cannot be read (Rulings 40, 41) ------------------------------------------------------
# The name of a raw file is the match key and its text holds the players' identifiers. An exception that comes out of ``open``,
# ``Path.is_file`` or ``json`` can carry them: an OSError names the path (in its text, its traceback, ``filename`` and a pickle), a
# UnicodeDecodeError shows the whole text in its ``repr`` and ``args``, a JSONDecodeError keeps it in ``doc``. A pool of S2 workers sends
# its exceptions to the parent as pickles, so what an exception carries counts as much as what it prints. FILE_KEY stands for a match
# key in the names and in the text of the files; ``synth.IDENTIFIER_SENTINELS`` for the identifiers in their text.
FILE_KEY = 'SENTINEL-MATCH-KEY-4711'
PATCH = '15.18'


def surfaces(exc: BaseException) -> dict:
    """Everything an exception shows or carries, by name: its text, repr and args, the attributes that can hold a path or a text
    (``filename``, ``doc``, ``object`` and the rest of ``vars``), the same three after a pickle round trip, the formatted traceback with
    its chain, and the bytes of the pickle."""
    blob = pickle.dumps(exc)
    out = {'traceback': ''.join(traceback.format_exception(exc)), 'pickle': blob.decode('latin-1')}
    for label, e in (('', exc), ('clone ', pickle.loads(blob))):
        held = [getattr(e, name, None) for name in ('filename', 'filename2', 'doc', 'object')] + [vars(e)]
        out |= {label + 'str': str(e), label + 'repr': repr(e), label + 'args': repr(e.args), label + 'attributes': repr(held)}
    return out


def leaks(exc: BaseException, *needles: str) -> dict:
    """``{surface: [needles it holds]}`` for the surfaces of ``exc`` that hold any of the ``needles``; ``{}`` if none does. The needles
    are sentinels and paths of the test, so a failing assert may print them; it never prints the exception."""
    found = {name: [n for n in needles if n in text] for name, text in surfaces(exc).items()}
    return {name: hits for name, hits in found.items() if hits}


def write_keyed_match(root) -> None:
    """The two files of a synthetic match whose key is FILE_KEY, under ``root``."""
    rec = synth.make_record(T_ms=400_000, key=FILE_KEY, patch=PATCH)
    synth.write_raw(root, PATCH, FILE_KEY, synth.raw_timeline(rec), synth.raw_detail(rec))


def keyed_file(root, kind: str) -> Path:
    """The path of the ``kind`` ('timeline' or 'detail') file that ``write_keyed_match`` wrote."""
    return Path(root) / PATCH / 'raw' / C.RAW_REGION[PATCH] / kind / f'{FILE_KEY}.json'


def failing_open(kind: str, make_error):
    """An ``open`` for the reader that raises ``make_error(path)`` for the file of that kind (the folder it lies in) and opens any other."""
    def fake(path, *args, **kwargs):
        if Path(path).parent.name == kind:
            raise make_error(path)
        return open(path, *args, **kwargs)
    return fake


def permission_denied(path):
    return PermissionError(errno.EACCES, 'Permission denied', str(path))


def locked_by_another_process(path):
    return PermissionError(errno.EACCES, 'The process cannot access the file because it is being used by another process', str(path))


def io_error(path):
    return OSError(errno.EIO, 'Input/output error', str(path))


READ_FAILURES = [('detail', permission_denied, 'PermissionError', errno.EACCES),
                 ('timeline', permission_denied, 'PermissionError', errno.EACCES),
                 ('detail', locked_by_another_process, 'PermissionError', errno.EACCES),
                 ('timeline', io_error, 'OSError', errno.EIO)]


@pytest.mark.parametrize('kind, make_error, name, number', READ_FAILURES,
                         ids=['detail-denied', 'timeline-denied', 'detail-locked', 'timeline-io'])
def test_a_file_that_cannot_be_opened_raises_a_key_free_error(tmp_path, monkeypatch, kind, make_error, name, number):
    """The native OSError names the file, so the match key; its text, its traceback, ``filename`` and a pickle all hold it. The reader
    replaces it with a ``MatchReadError`` that says which kind of file, the class and the errno of the native error and the patch, and
    that has neither a cause nor a context (both would still hold the path)."""
    write_keyed_match(tmp_path)
    monkeypatch.setattr(reader, 'open', failing_open(kind, make_error), raising=False)
    with pytest.raises(OSError) as err:
        reader.read_match(FILE_KEY, PATCH, tmp_path)
    exc = err.value
    assert leaks(exc, FILE_KEY, str(tmp_path)) == {}                                         # nothing shows the key or the path ...
    assert type(exc) is reader.MatchReadError                                                # ... and the error is the reader's own
    assert str(exc) == f'{kind} file cannot be read: {name}, errno {number} (patch {PATCH})'
    assert (exc.errno, exc.__cause__, exc.__context__) == (number, None, None)
    assert FILE_KEY.encode() not in pickle.dumps(exc)
    clone = pickle.loads(pickle.dumps(exc))                                                  # it must survive a worker pool
    assert type(clone) is reader.MatchReadError and (str(clone), clone.errno) == (str(exc), number)


def test_an_error_while_the_file_is_read_is_key_free_too(tmp_path, monkeypatch):
    """The OSError of ``read`` has no path, but it is the same trouble and it ends the same way."""
    class Unreadable:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            raise OSError(errno.EIO, 'Input/output error')
    write_keyed_match(tmp_path)
    monkeypatch.setattr(reader, 'open', lambda path, *args, **kwargs: Unreadable(), raising=False)
    with pytest.raises(OSError) as err:
        reader.read_match(FILE_KEY, PATCH, tmp_path)
    exc = err.value
    assert leaks(exc, FILE_KEY) == {} and type(exc) is reader.MatchReadError
    assert str(exc) == f'detail file cannot be read: OSError, errno {errno.EIO} (patch {PATCH})'      # the detail file is read first


def test_a_file_that_cannot_even_be_looked_up_raises_the_same_key_free_error(tmp_path, monkeypatch):
    """``Path.is_file`` answers False for a missing file but raises a permission error, and that error names the path."""
    write_keyed_match(tmp_path)
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == f'{FILE_KEY}.json':
            raise PermissionError(errno.EACCES, 'Access is denied', str(self))
        return real_stat(self, *args, **kwargs)
    with monkeypatch.context() as patched:
        patched.setattr(Path, 'stat', stat)
        with pytest.raises(OSError) as err:
            reader.read_match(FILE_KEY, PATCH, tmp_path)
    exc = err.value
    assert leaks(exc, FILE_KEY, str(tmp_path)) == {}
    assert type(exc) is reader.MatchReadError
    assert str(exc) == f'timeline file cannot be read: PermissionError, errno {errno.EACCES} (patch {PATCH})'
    assert (exc.__cause__, exc.__context__) == (None, None)


def test_opening_a_folder_in_place_of_a_file_is_key_free_with_no_stand_in_for_open(tmp_path, monkeypatch):
    """A real OSError from the operating system: both paths are folders that are named like match files (``is_file`` is told they are
    files, so that the reader goes on to open them)."""
    for kind in ('timeline', 'detail'):
        keyed_file(tmp_path, kind).mkdir(parents=True)
    try:
        open(keyed_file(tmp_path, 'detail'), 'rb')
    except OSError as native:
        name, number = type(native).__name__, native.errno
    with monkeypatch.context() as patched:
        patched.setattr(Path, 'is_file', lambda self, **kwargs: True)
        with pytest.raises(OSError) as err:
            reader.read_match(FILE_KEY, PATCH, tmp_path)
    exc = err.value
    assert leaks(exc, FILE_KEY, str(tmp_path)) == {}
    assert type(exc) is reader.MatchReadError
    assert str(exc) == f'detail file cannot be read: {name}, errno {number} (patch {PATCH})'


def test_the_read_error_is_an_os_error_and_neither_a_rejection_nor_a_format_error():
    """It is an error that stops the run, as the native OSError did: a ``MatchRejected`` would be counted and skipped."""
    assert issubclass(reader.MatchReadError, OSError)
    assert not issubclass(reader.MatchReadError, (reader.MatchRejected, ValueError))


def test_a_read_error_stops_the_s1_extraction_instead_of_becoming_a_counted_rejection(tmp_path, monkeypatch):
    from c_ept import run_s1                  # here: this file needs the S1 runner chain for nothing else
    write_keyed_match(tmp_path)
    monkeypatch.setattr(reader, 'open', failing_open('detail', permission_denied), raising=False)
    with pytest.raises(reader.MatchReadError) as err:           # a MatchRejected would come back as {'reject': code}
        run_s1.extract_match(PATCH, FILE_KEY, 'FIT', lambda key, patch: reader.read_match(key, patch, tmp_path))
    assert leaks(err.value, FILE_KEY) == {}


DAMAGE = {          # case: how the bytes of a good file are damaged
    'cut_in_half': lambda good: good[:len(good) // 2],
    'empty': lambda good: b'',
    'bad_utf8_byte': lambda good: good[:150] + b'\xff' + good[150:],
    'cut_in_a_multibyte_character': lambda good: good[:200] + b'\xed\x95',         # the first two of the three bytes of a Korean syllable
    'zero_filled': lambda good: b'\x00' * 64 + good,                                # json takes a file that starts with NUL for UTF-32
}


@pytest.mark.parametrize('kind', ['detail', 'timeline'])
@pytest.mark.parametrize('case', list(DAMAGE))
def test_a_file_that_is_not_utf8_json_raises_a_format_error_without_its_text(tmp_path, kind, case):
    """A UnicodeDecodeError shows the whole file in its repr and args and a JSONDecodeError keeps it in ``doc``: the text holds the match
    key, the puuids, riot ids and summoner names. The reader names the kind of file, the class and the offset of the native error and
    the patch in a ``MatchFormatError`` (a ValueError, as the native errors are; the key stays an attribute, as in every one of them)."""
    write_keyed_match(tmp_path)
    path = keyed_file(tmp_path, kind)
    bad = DAMAGE[case](path.read_bytes())
    path.write_bytes(bad)
    try:
        json.loads(bad)
    except ValueError as native:                                       # what json says about the bytes: the reader may name this much
        name = type(native).__name__
        offset = native.pos if isinstance(native, json.JSONDecodeError) else native.start
    with pytest.raises(ValueError) as err:
        reader.read_match(FILE_KEY, PATCH, tmp_path)
    exc = err.value
    assert leaks(exc, str(tmp_path), *synth.IDENTIFIER_SENTINELS) == {}                      # nothing of the text of the file
    assert set(leaks(exc, FILE_KEY)) <= {'attributes', 'clone attributes', 'pickle'}          # the key: an attribute, not in any text
    assert type(exc) is reader.MatchFormatError and (exc.key, exc.patch) == (FILE_KEY, PATCH)
    assert str(exc) == f'the {kind} file is not valid UTF-8 JSON: {name} at offset {offset} (patch {PATCH})'
    assert (exc.__cause__, exc.__context__) == (None, None)
