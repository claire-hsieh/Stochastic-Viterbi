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

def stochastic_viterbi(transition_file, seq, tr):
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

        # traceback
        multiple_traceback(trellis, state, paths, tr)
        # elif tr == 1: # multiple traceback
            # loop through all 
            # for i in range(len(trellis)):
                

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

def multiple_traceback(trellis, state, paths, tr):
    # tr = number of tracebacks
    temp = dict((key, {}) for key in state.keys())
    path = []
    for p in range(tr):
        max_state = random.choice(list(state.keys()))
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

def forward_backward(json_file, seq, log=1):
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
                    fw = emission[trans][o] * state[trans]
            else:
                for prev in transition[trans]:
                    if transition[prev][trans] != 0:
                        fw += forward[ind-1][prev] * transition[prev][trans] * emission[trans][o]
            forward[ind][trans] = fw
        # normalize
        tot = sum(forward[ind].values())
        for t in forward[ind]: 
            forward[ind][t] /= tot
    forward.insert(0, dict((key, state[key]) for key in transition.keys()))

    # Compute backward probability
    backward = list(dict((key, 0) for key in transition.keys()) for i in range(len(seq)))
    backward.append(dict((key, 1) for key in transition.keys()))
    for o, ind in zip(seq[::-1], reversed(range(len(seq)))):
        for trans in transition:
            bw = 0
            for next in transition[trans]:
                if transition[next][trans] != 0:
                    # print(ind, o, trans, next, backward[ind+1][next], transition[trans][next], emission[next][o])
                    bw += backward[ind+1][next] * transition[next][trans] * emission[next][o]
            backward[ind][trans] = bw
        # normalize
        tot = sum(backward[ind].values())
        for t in backward[ind]:
            backward[ind][t] /= tot

    # Compute smoothed values
    smoothed = list(dict((key, 0) for key in transition.keys()) for i in range(len(seq)))
    smoothed.insert(0, dict((key, backward[0][key] * forward[0][key]) for key in transition.keys()))
    for ind in range(len(forward)):
        for trans in transition.keys():
            smoothed[ind][trans] = backward[ind][trans] * forward[ind][trans]
        # normalize
        tot = sum(smoothed[ind].values())
        for t in smoothed[ind]:
            smoothed[ind][t] /= tot
    
    return forward, backward, smoothed


### FORMAT OF TRELLIS ###
# trellis = [{initial_state: log_probability}, {current_state: {previous_state: log_probability}, ... }]

### TO DO ###
# calculate probability of path 
# apply forward backward to path (seperate)
# do MULTIPLE traceback: add random element 
# figure out way to do documentation