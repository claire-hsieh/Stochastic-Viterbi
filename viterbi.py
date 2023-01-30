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

def viterbi(state, transition, transition_states, emission, emission_states, seq, log=0):
    # input: list of state, transition, and emission prob as lists
    # state should be in format: [[p1, p2, ...]]
    # transition indices: list of states
    state, transition, emission = format_prob(state, transition, transition_states, emission, emission_states)
    
    if transition.shape[1]!= emission.shape[0]: 
        # num rows of trans. states must equal num col. of emission states
        print("Number of rows in transition state must equal number of columns in emission states", file=sys.stderr)
    probability = 1
    path = []
    if type(seq) == str:
        seq = [*seq]
    for s,i in zip(seq, range(len(seq))):
        if i == 0: # if first in seq, use state prob.
            prob1 = {}
            for st in state:
                prob1[st] = float(state[st] * emission[s][st]) 
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
                    prob[t] = prev_prob +  math.log2(transition[prev_state][t] * emission[s][t])
                else:
                    prob[t] = prev_prob * (transition[prev_state][t] * emission[s][t])
            x = max(prob, key=prob.get)
            path.append(x)
            prev_prob = prob[x]
            prev_state = x
    return path, prev_prob



"""
SAMPLE INPUT

state = [[0.5, 0.5]]
transition = [[0.5,0.4], [0.4,0.6]] 
transition_states = ['H', 'L']
emission = [[0.2,0.3,0.3,0.2], [0.3,0.2,0.2,0.3]] # hidden states, row: A,C,G,T, col: H, L
emission_states = ['A', 'C', 'G', 'T']
seq = 'GGCACTGAA'

viterbi(state, transition, transition_states, emission, emission_states, seq)

"""