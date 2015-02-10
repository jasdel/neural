package neural

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestFeedForwardCreation(t *testing.T) {
	net := NewFeedForward(2, 2, []int{1}, PerceptronActivation, SumInputs)

	assert.Len(t, net.Inputs, 2, "Expect two inputs")
	require.Equal(t, 1, len(net.HiddenLayers), "Expect single hidden layer")
	assert.Len(t, net.HiddenLayers[0].Nodes, 1, "Expect single hidden node in layer")
	assert.Len(t, net.Outputs.Nodes, 2, "Expect two output nodes")

	assert.Exactly(t, net.HiddenLayers[0].Inputs, net.HiddenLayers[0].Nodes[0].Inputs,
		"Expect layer inputs to be referenced by nodes")
	assert.Exactly(t, net.Outputs.Inputs, net.Outputs.Nodes[0].Inputs,
		"Expect layer inputs to be referenced by nodes")
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
