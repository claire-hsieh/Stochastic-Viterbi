# Viterbi Algorithm

A simple viterbi decoding algorithm implementation in python.

## Usage

Transition, emission, and state probabilities are defined in a json file. The format of the file is as follows:

```json
{
    "states": ["H", "C"],
    "start_prob": {
        "H": 0.6,
        "C": 0.4
    },
    "transition_prob": {
        "H": {
            "H": 0.7,
            "C": 0.3
        },
        "C": {
            "H": 0.4,
            "C": 0.6
        }
    },
    "emission_prob": {
        "H": {
            "1": 0.1,
            "2": 0.4,
            "3": 0.5
        },
        "C": {
            "1": 0.7,
            "2": 0.2,
            "3": 0.1
        }
    }
}
```

To run the algorithm, simply run the following command:

`viterbi(transition_file, sequence)`

Where `transition_file` is the path to the json file containing the transition, emission, and state probabilities, and `sequence` is the sequence of observations.
