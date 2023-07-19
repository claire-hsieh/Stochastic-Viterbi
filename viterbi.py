import numpy as np
import pandas as pd
import math

def format_prob(state, transition, transition_states, emission, emission_states):
    # input prob as list of lists and states as list of states
    # output: pandas dataframe with states as indices
    df_state = pd.DataFrame(state, columns = transition_states)
    df_trans = pd.DataFrame(transition, index = transition_states, columns = transition_states)
    df_em = pd.DataFrame(emission, index = transition_states, columns = emission_states,)
    return df_state, df_trans, df_em

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
