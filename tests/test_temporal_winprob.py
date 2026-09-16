import numpy as np
import pytest

from train.temporal_winprob import (random_minute_index, feature_matrix, observed_queries,
                                    maymin_model, calibrated_variants)


def test_sampling_is_match_deterministic_and_independent_of_outcomes():
    a = [random_minute_index(f'm{i}', 30) for i in range(200)]
    b = {f'm{i}': random_minute_index(f'm{i}', 30) for i in reversed(range(200))}
    assert a == [b[f'm{i}'] for i in range(200)]
    assert len(set(a)) == 30
    with pytest.raises(ValueError):
        random_minute_index('m', 0)


def test_maymin_seven_inputs_aggregate_objectives_once():
    from tests.test_state_value import fixture_pack
    from gameplay.state_value import StateBuilder
    names, pack = fixture_pack()
    values = StateBuilder(pack,names).at(60000).values
    values.update(blue_dragons=3.,blue_baron=1.,blue_elder=1.,blue_herald=2.,blue_horde=6.,blue_atakhan=1.,
                  blue_tower_OUTER_TURRET=3.,blue_tower_INNER_TURRET=2.)
    X, names = feature_matrix(np.array([list(values.values())]),list(values),'maymin')
    assert X.shape == (1,7)
    assert X[0,names.index('blue_monsters')] == 14
    assert X[0,names.index('blue_towers')] == 5


def test_immediate_proxy_excludes_future_and_terminal_followup():
    assert observed_queries(100, 150, 50000, 50000) == [99,150,30150,-1]
    with pytest.raises(ValueError):
        observed_queries(100,1000,1000,2000)
    with pytest.raises(ValueError):
        observed_queries(100,90,1000,2000)


def test_snapshot_clock_age_cannot_drive_expanded_value():
    names=['time_minutes','snapshot_age_s','blue_baron']
    states=np.array([[10.,0.,1.],[10.,59.9,1.]])
    X, columns=feature_matrix(states,names,'expanded')
    assert 'snapshot_age_s' not in columns
    assert np.array_equal(X[0],X[1])


def test_calibration_outputs_remain_probabilities():
    # Seven explicit raw columns are enough to exercise inference/calibration.
    names = ['time_minutes','red_kills','blue_kills']
    names += [t+k for k in ['tower_'+s for s in ('OUTER_TURRET','INNER_TURRET','BASE_TURRET','NEXUS_TURRET','OTHER')]
              +['dragons','baron','elder','herald','horde','atakhan'] for t in ('red_','blue_')]
    X = np.random.default_rng(7).normal(size=(100,len(names)))
    y = (X[:,1] > 0).astype(int)
    features,_ = feature_matrix(X,names,'maymin')
    base = maymin_model().fit(features[:60],y[:60])
    models = calibrated_variants('maymin',names,base,X[60:80],y[60:80])
    for model in models.values():
        p = model.predict_proba(X[80:])
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.allclose(p.sum(axis=1),1)
