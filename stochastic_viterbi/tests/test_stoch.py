import math
import stochastic_viterbi
import pytest
import json
json_file = "./tests/stoch_hmm.json"
seq = "umbrella, umbrella, no_umbrella, umbrella, umbrella"
seq = seq.split(", ")
forward_ts, backward_ts, smoothed_ts = stochastic_viterbi.forward_backward(json_file, seq)

def test_forward():
    vals = [0.5, 0.5, 0.8182, 0.181, 0.8834, 0.1166, 0.1907, 0.8093, 0.7308, 0.2692, 0.8673, 0.1327]
    correct = list(dict((key, vals[i*2+j]) for j, key in enumerate(["rain", "no_rain"])) for i in range(len(seq)))
    for i, j in zip(correct, forward_ts):
        for k in ["rain", "no_rain"]:
            assert i[k] == pytest.approx(j[k], abs=1e-3)

def test_backward():
    vals = [0.6469, 0.3531, 0.5923, 0.4077, 0.3763, 0.6237, 0.6533, 0.3467, 0.6273, 0.3727, 1.0, 1.0]
    correct = list(dict((key, vals[i*2+j]) for j, key in enumerate(["rain", "no_rain"])) for i in range(len(seq)+1))
    for i, j in zip(correct, backward_ts):
        for k in ["rain", "no_rain"]:
            assert i[k] == pytest.approx(j[k], abs=1e-3)

def test_smoothed():
    vals = [0.6469, 0.3531, 0.8673, 0.1327, 0.8204, 0.1796, 0.3075, 0.6925, 0.8204, 0.1796, 0.8673, 0.1327]
    correct = list(dict((key, vals[i*2+j]) for j, key in enumerate(["rain", "no_rain"])) for i in range(len(seq)))
    for i, j in zip(correct, smoothed_ts):
        for k in ["rain", "no_rain"]:
            assert i[k] == pytest.approx(j[k], abs=1e-3)