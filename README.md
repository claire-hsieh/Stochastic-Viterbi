# viterbi algorithm

## SAMPLE INPUT <br>
state = [[0.5, 0.5]] <br>
transition = [[0.5,0.4], [0.4,0.6]] <br>
transition_states = ['H', 'L'] <br>
emission = [[0.2,0.3,0.3,0.2], [0.3,0.2,0.2,0.3]] # hidden states, row: A,C,G,T, col: H, L <br>
emission_states = ['A', 'C', 'G', 'T'] <br>
seq = 'GGCACTGAA' <br>
viterbi(state, transition, transition_states, emission, emission_states, seq) <br>

## SAMPLE OUTPUT <br>
(['Healthy', 'Healthy', 'Fever'], -11.772868005544115)
