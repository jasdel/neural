package neural

import (
	"testing"
)

type percpInput struct {
	A, B, Out float64
}

type percpTestCase struct {
	Desc       string
	Weights    []float64
	Threshold  float64
	Activation ActivationFunction
	InputSets  []percpInput
}

var percpTestCases = []percpTestCase{
	percpTestCase{
		Desc:       "OR",
		Weights:    []float64{0, 1, 1},
		Threshold:  1,
		Activation: PerceptronActivation,
		InputSets: []percpInput{
			percpInput{A: 0, B: 0, Out: 0},
			percpInput{A: 1, B: 0, Out: 1},
			percpInput{A: 0, B: 1, Out: 1},
			percpInput{A: 1, B: 1, Out: 1},
		},
	},
	percpTestCase{
		Desc:       "AND",
		Weights:    []float64{0, 0.5, 0.5},
		Threshold:  1,
		Activation: PerceptronActivation,
		InputSets: []percpInput{
			percpInput{A: 0, B: 0, Out: 0},
			percpInput{A: 1, B: 0, Out: 0},
			percpInput{A: 0, B: 1, Out: 0},
			percpInput{A: 1, B: 1, Out: 1},
		},
	},
	percpTestCase{
		Desc:       "A AND NOT B",
		Weights:    []float64{0, 1, -1},
		Threshold:  1,
		Activation: PerceptronActivation,
		InputSets: []percpInput{
			percpInput{A: 0, B: 0, Out: 0},
			percpInput{A: 1, B: 0, Out: 1},
			percpInput{A: 0, B: 1, Out: 0},
			percpInput{A: 1, B: 1, Out: 0},
		},
	},
}

func TestPerceptron(t *testing.T) {
	for i, c := range percpTestCases {
		p := buildPerceptron(c.Weights, c.Threshold, c.Activation)
		for j, in := range c.InputSets {
			p.Inputs[1] = in.A
			p.Inputs[2] = in.B
			if o := p.Compute(); o != in.Out {
				t.Errorf("Unexpected output of case[%d] %s for input[%d](%f.%f) got: %f, expected %f",
					i, c.Desc, j, in.A, in.B, o, in.Out)
			}
		}
	}
}

func TestPerceptronXOR(t *testing.T) {
	p1 := buildPerceptron([]float64{0, 0.5, 0.5}, 1, PerceptronActivation)
	p2 := buildPerceptron([]float64{0, 1, 1, -2}, 1, PerceptronActivation)

	// 0,0 = 0
	p1.Inputs[1] = 0
	p1.Inputs[2] = 0
	p2.Inputs[1] = p1.Inputs[1]
	p2.Inputs[2] = p1.Inputs[2]
	p2.Inputs[3] = p1.Compute()
	if o := p2.Compute(); o != 0 {
		t.Errorf("Unexpected output for XOR 0,0=0, %f", o)
	}
	// 0,1 = 1
	p1.Inputs[1] = 0
	p1.Inputs[2] = 1
	p2.Inputs[1] = p1.Inputs[1]
	p2.Inputs[2] = p1.Inputs[2]
	p2.Inputs[3] = p1.Compute()
	if o := p2.Compute(); o != 1 {
		t.Errorf("Unexpected output for XOR 0,1=1, %f", o)
	}
	// 1,0 = 1
	p1.Inputs[1] = 1
	p1.Inputs[2] = 0
	p2.Inputs[1] = p1.Inputs[1]
	p2.Inputs[2] = p1.Inputs[2]
	p2.Inputs[3] = p1.Compute()
	if o := p2.Compute(); o != 1 {
		t.Errorf("Unexpected output for XOR 1,0=1, %f", o)
	}
	// 1,1 = 0
	p1.Inputs[1] = 1
	p1.Inputs[2] = 1
	p2.Inputs[1] = p1.Inputs[1]
	p2.Inputs[2] = p1.Inputs[2]
	p2.Inputs[3] = p1.Compute()
	if o := p2.Compute(); o != 0 {
		t.Errorf("Unexpected output for XOR 1,1=0, %f", o)
	}
}

func buildPerceptron(w []float64, t float64, actFunc ActivationFunction) *Node {
	inputs := make([]float64, len(w))
	return &Node{
		Inputs:     inputs,
		Weights:    w,
		Threshold:  t,
		Activation: actFunc,
	}
}
