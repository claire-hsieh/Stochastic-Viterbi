# Viterbi Algorithm

A simple viterbi decoding algorithm implementation in python.

## Usage

Transition, emission, and state probabilities are defined in a json file. The format of the file is as follows:

```json
{
	"states": 2,
	"state": [
		{
			"name": "rain",
			"init": 0.5,
			"term": 0.0,
			"transitions": 2,
			"transition": {
				"rain": 0.7,
				"no_rain": 0.3
			},
			"emissions": 2,
			"emission": {"umbrella": 0.9, "no_umbrella": 0.1}
		},
		{
			"name": "no_rain",
			"init": 0.5,
			"term": 0.0,
			"transitions": 2,
			"transition": {
				"rain": 0.3,
				"no_rain": 0.7
			},
			"emissions": 2,
			"emission": {"umbrella": 0.2, "no_umbrella": 0.8}
		}
	]
}
```

To run the algorithm, simply run the following command:

`viterbi(transition_file, sequence)`

Where `transition_file` is the path to the json file containing the transition, emission, and state probabilities, and `sequence` is the sequence of observations.
