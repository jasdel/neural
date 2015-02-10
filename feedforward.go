package neural

import (
	"log"
)

type Layer struct {
	Nodes  []*Node
	Inputs []float64
}

type FeedForward struct {
	Inputs       []float64
	HiddenLayers []*Layer
	Outputs      *Layer
}

func NewFeedForward(inputs, outputs int, hiddenLayers []int, hiddenAct, outAct ActivationFunction) *FeedForward {
	net := &FeedForward{
		Inputs:       make([]float64, inputs),
		HiddenLayers: make([]*Layer, len(hiddenLayers)),
		Outputs:      &Layer{Nodes: make([]*Node, outputs)},
	}

	prevNodes := inputs
	for i := 0; i < len(hiddenLayers); i++ {
		hidden := hiddenLayers[i]
		layer := &Layer{Nodes: make([]*Node, hidden)}
		layer.Inputs = make([]float64, prevNodes)
		for j := 0; j < hidden; j++ {
			layer.Nodes[j] = &Node{
				Inputs:     layer.Inputs,
				Weights:    make([]float64, prevNodes),
				Activation: hiddenAct,
			}
		}
		net.HiddenLayers[i] = layer
		prevNodes = hidden
	}

	net.Outputs.Inputs = make([]float64, prevNodes)
	for i := 0; i < outputs; i++ {
		net.Outputs.Nodes[i] = &Node{
			Inputs:     net.Outputs.Inputs,
			Weights:    make([]float64, prevNodes),
			Activation: outAct,
		}
	}

	return net
}

// Push the inputs into the neural network. Returning the
func (f *FeedForward) Forward(inputs []float64) []float64 {
	if len(inputs) != len(f.Inputs) {
		log.Println("FeedForward.PushInputs failed input lengths do not match", len(inputs), len(f.Inputs))
		return nil
	}

	copy(f.Inputs, inputs)

	toProp := f.Inputs
	for i := 0; i < len(f.HiddenLayers); i++ {
		toProp = propagateInputs(toProp, f.HiddenLayers[i])
	}
	outputs := propagateInputs(toProp, f.Outputs)

	return outputs
}

// Adds inputs to a layer, computes each node of that layer,
// SKips the first input of the node because it is the bias.
func propagateInputs(inputs []float64, layer *Layer) []float64 {
	outputs := make([]float64, len(layer.Nodes))
	copy(layer.Inputs[1:], inputs)
	for i := 0; i < len(layer.Nodes); i++ {
		outputs[i] = layer.Nodes[i].Compute()
	}
	return outputs
}
