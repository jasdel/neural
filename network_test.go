package neural

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math/rand"
	"testing"
)

type stubNetworkLearner struct{}

func TestNetworkCreation(t *testing.T) {
	net := NewNetwork(2, 2, []int{1}, PerceptronActivation, SumInputs)

	assert.Len(t, net.Inputs, 2, "Expect two inputs")
	require.Equal(t, 2, len(net.Layers), "Expect two layers, 1 hidden 1 output")
	assert.Len(t, net.Layers[0].Nodes, 1, "Expect single hidden node in layer")
	assert.Len(t, net.Layers[1].Nodes, 2, "Expect two output nodes")

	assert.Exactly(t, net.Layers[0].Inputs, net.Layers[0].Nodes[0].Inputs,
		"Expect layer inputs to be referenced by nodes")
	assert.Exactly(t, net.Layers[1].Inputs, net.Layers[1].Nodes[0].Inputs,
		"Expect layer inputs to be referenced by nodes")
}

func TestRandomizeWeights(t *testing.T) {
	net := NewNetwork(2, 1, []int{1}, SigmoidActivation, SumInputs)

	low := -0.05
	high := 0.05
	net.RandomizeWeights(low, high, rand.NewSource(1))

	for i, layer := range net.Layers {
		for j, node := range layer.Nodes {
			for k, weight := range node.Weights {
				if weight < low || weight > high {
					t.Errorf("Unexpected weight, %d,%d,%d, %f", i, j, k, weight)
				}
			}
		}
	}
}

func TestPropagateInputs(t *testing.T) {
	inputs := []float64{0, 1, 0}
	node := buildPerceptron([]float64{0, 0.5, 0.5}, 1, PerceptronActivation)
	node.Inputs = inputs
	layer := &Layer{
		Inputs: inputs,
		Nodes:  []*Node{node},
	}

	outputs := propagateInputs([]float64{0, 0}, layer)
	require.Equal(t, 1, len(outputs), "Expected only a single output")
	assert.Equal(t, 0, outputs[0], "Expected AND 0,0=0, %f", outputs[0])

	outputs = propagateInputs([]float64{0, 1}, layer)
	assert.Equal(t, 0, outputs[0], "Expected AND 0,1=0, %f", outputs[0])

	outputs = propagateInputs([]float64{1, 0}, layer)
	assert.Equal(t, 0, outputs[0], "Expected AND 1,0=0, %f", outputs[0])

	outputs = propagateInputs([]float64{1, 1}, layer)
	assert.Equal(t, 1, outputs[0], "Expected AND 1,1=1, %f", outputs[0])
}
