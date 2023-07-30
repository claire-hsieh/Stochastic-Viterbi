import math
import argparse
import sys
import gzip
import json

# parser = argparse.ArgumentParser(
# 	description='General Stochastic Viterbi Algorithm')
# parser.add_argument('-seq', type=list, required=True,
# 	default=[], help = 'sequence')
# parser.add_argument('jsonhmm', type=str, required=True, help="json file of transition and emission probabilities")
# arg = parser.parse_args()

def viterbi(transition_file, seq, log=1):
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
        probability = 1
        path = []
        if type(seq) == str:
            seq = [*seq]
        for s,i in zip(seq, range(len(seq))):
            if i == 0: # if first in seq, use state prob.
                prob1 = {}
                for st in state:
                    prob1[st] = float(state[st] * emission[st][s]) 
                t_state = max(prob1, key=prob1.get)
                path.append(t_state) # add max probability to path
                if log == 1:
                    prev_prob = math.log2(prob1[path[0]])
                else:
                    prev_prob = prob1[path[0]]
                prev_state = t_state
            else:
                prob = {}           
                for t in transition:      
                    if log == 1:
                        prob[t] = prev_prob +  math.log2(transition[prev_state][t] * emission[t][s])
                    else:
                        prob[t] = prev_prob * (transition[prev_state][t] * emission[t][s])
                x = max(prob, key=prob.get)
                path.append(x)
                prev_prob = prob[x]
                prev_state = x
    return path, prev_prob



def stochastic_viterbi(json_file, seq, log=1):    
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
        probability = 1
        path = []
        if type(seq) == str:
            seq = [*seq]
        for s,i in zip(seq, range(len(seq))):
            if i == 0: # if first in seq, use state prob.
                prob1 = dict((key, 0) for key in transition.keys())
                for trans in transition.keys():
                    if state[trans] != 0 and emission[trans][s] != 0: 
                        prob1[trans] = math.log(state[trans]) + math.log(emission[trans][s])
            else:
                prob1 = dict((key, dict((key, 0) for key in transition.keys())) for key in transition.keys())
                for trans1 in path[i-1].keys(): # iterate through previous states
                    if i == 1: 
                        max_prev = path[i-1][max(path[i-1], key= lambda x: path[i-1][x])]
                    else: 
                        max_prev = path[i-1][trans1][max(path[i-1][trans1], key= lambda x: path[i-1][trans1][x])]
                    for trans2 in transition[trans1].keys(): # iterate through current states
                        # print(trans1, trans2, max_prev, path[i-1][max_prev])#, math.log(transition[trans1][trans2])), math.log(emission[trans2][s]))
                        prob1[trans2][trans1] = max_prev + max_prev + math.log(transition[trans1][trans2]) + math.log(emission[trans2][s])
            path.append(prob1)
    return path

def forward_backward(json_file, seq, log=1):
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
    
    # Traceback (smoothed values)

    
    return forward, backward, smoothed

