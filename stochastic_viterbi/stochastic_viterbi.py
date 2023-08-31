import math
import argparse
import json
import random

# parser = argparse.ArgumentParser(
# 	description='General Stochastic Viterbi Algorithm')
# parser.add_argument('-seq', type=list, required=True,
# 	default=[], help = 'sequence')
# parser.add_argument('jsonhmm', type=str, required=True, help="json file of transition and emission probabilities")
# arg = parser.parse_args()

def stochastic_viterbi(transition_file, seq, illegal, trials=100):
    # illegal: list of illegal states to end in 
    state = {}
    transition = {}
    emission = {}
    with open(transition_file) as f:
        x = json.load(f)
        for i in range(x['states']):
            temp = x['state'][i]
            state[temp['name']] = temp['init']
            transition[temp['name']] = temp['transition']
            emission[temp['name']] = temp['emission']
        if len(transition) != len(emission):
            raise ValueError("Transition and emission matrices must be the same length")
        trellis = []
        paths = []
        if type(seq) == str:
            seq = [*seq]
        # fill out viterbi trellis
        for s,i in zip(seq, range(len(seq))):
            if i == 0: # if first in seq, use state prob.
                prob1 = dict((key, 0) for key in state.keys())
                for trans in transition.keys():
                    if state[trans] != 0 and emission[trans][s] != 0: 
                        prob1[trans] = math.log(state[trans]) + math.log(emission[trans][s])
            else:
                prob1 = dict((key1, dict((key2, 0) for key2 in transition[key1].keys())) for key1 in transition.keys())
                for trans1 in trellis[i-1].keys(): # iterate through previous states
                    if i == 1: 
                        max_prev = trellis[i-1][max(trellis[i-1], key= lambda x: trellis[i-1][x])]
                    else: 
                        max_prev = trellis[i-1][trans1][max(trellis[i-1][trans1], key= lambda x: trellis[i-1][trans1][x])]
                    for trans2 in prob1[trans1].keys(): # iterate through current states
                        # print(trans1, trans2, max_prev, path[i-1][max_prev])#, math.log(transition[trans1][trans2])), math.log(emission[trans2][s]))
                        prob1[trans1][trans2] = max_prev + max_prev + math.log(transition[trans1][trans2]) + math.log(emission[trans2][s])
            trellis.append(prob1)       
        multiple_traceback(trellis, state, paths, illegal, trials)
        return trellis, paths

def traceback(trellis, state, path):
    temp = dict((key, {}) for key in state.keys())
    for i in range(len(trellis)-1, 1, -1):
        for trans1 in trellis[i]: # loop through last state
            for trans4 in trellis[i-1]: # loop through second to last state
                max_state = max(trellis[i-1][trans4], key= lambda x: trellis[i-1][trans4][x])    
                temp[trans4][max_state] = trellis[-1][trans4][max_state]
        max_state = max(temp, key= lambda x: list(temp[x].values())[0])
        path.append(list(temp[max_state].keys())[0]) # add last state to path
        path.append(max_state) # add second to last state to path      
    return path

def max_traceback(trellis, state, paths, illegal):
    temp = dict((key, {}) for key in state.keys())
    states = [i for i in state if i not in illegal]
    path = []
    for s in states:
        max_state = s
        for i in range(len(trellis)-2, 1, -1):
            for trans1 in trellis[i]: # loop through last state
                for trans4 in trellis[i-1]: # loop through second to last state
                    max_state = max(trellis[i-1][trans4], key= lambda x: trellis[i-1][trans4][x])    
                    temp[trans4][max_state] = trellis[-1][trans4][max_state]
        max_state = max(temp, key= lambda x: list(temp[x].values())[0])
        path.append(list(temp[max_state].keys())[0]) # add last state to path
        path.append(max_state) # add second to last state to path 
        paths.append(path) 
    return paths

def multiple_traceback(trellis, transition, paths, illegal, trials):
    states = [i for i in transition.keys() if i not in illegal]
    for t in range(trials):
        prev_state = random.choice(states)
        path = []
        for i in range(len(trellis)-1, -1, -1):
            possible_ts_states = [i for i in transition.keys() if prev_state in transition[i].keys()]
            probabilities = [transition[i][prev_state] for i in possible_ts_states]
            prev_state = random.choices(possible_ts_states, weights = probabilities, k=1)[0]
            path.append(prev_state) 
        paths.append(path) 
    return paths

def forward_backward(json_file, seq):
    state = {}
    transition = {}
    emission = {}
    with open(json_file) as f:
        x = json.load(f)
        for i in range(x['states']):
            temp = x['state'][i]
            state[temp['name']] = temp['init']
            transition[temp['name']] = temp['transition']
            emission[temp['name']] = temp['emission']
        if len(transition) != len(emission):
            raise ValueError("Transition and emission matrices must be the same length")
        
    # Compute forward probability
    forward = list(dict((key, 0) for key in transition.keys()) for i in range(len(seq)))
    for ind, o in enumerate(seq):
        for trans in transition:
            fw = 0
            if ind == 0:
                if not(emission[trans][o] == 0 or state[trans] == 0):
                    fw = math.log(emission[trans][o]) + math.log(state[trans])
            else:
                for prev in transition[trans]:
                    if trans in transition[prev]: # and transition[prev][trans] != 0:
                        fw += forward[ind-1][prev] + math.log(transition[prev][trans]) + math.log(emission[trans][o])
            forward[ind][trans] = fw
    forward.insert(0, dict((key, state[key]) for key in transition.keys()))

    # Compute backward probability
    backward = list(dict((key, 0) for key in transition.keys()) for i in range(len(seq)))
    backward.append(dict((key, 1) for key in transition.keys()))
    for o, ind in zip(seq[::-1], reversed(range(len(seq)))):
        for trans in transition:
            bw = 0
            for next in transition[trans]:
                if trans in transition[next]: #if transition[next][trans] != 0:
                    bw += backward[ind+1][next] + math.log(transition[next][trans]) + math.log(emission[next][o])
            backward[ind][trans] = bw


    # Compute smoothed values
    smoothed = list(dict((key, 0) for key in transition.keys()) for i in range(len(seq)))
    smoothed.insert(0, dict((key, backward[0][key] * forward[0][key]) for key in transition.keys()))
    for ind in range(len(forward)):
        for trans in transition.keys():
            smoothed[ind][trans] = backward[ind][trans] + forward[ind][trans]

    return smoothed


### FORMAT OF TRELLIS ###
# trellis = [{initial_state: log_probability}, {current_state: {previous_state: log_probability}, ... }]

### TO DO ###
# calculate probability of path 
# apply forward backward to path (seperate)
# do MULTIPLE traceback: add random element [DONE]
# figure out way to do documentation