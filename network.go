package neural

import (
	"fmt"
	"math"
	"math/rand"
)

// Hidden or output layer of a neural network
type Layer struct {
	Nodes  []*Node
	Inputs []float64
}

type Learner interface {
	Learn(outputs, targets []float64)
}

// Defines the feed forward neural network
// Layers includes all hidden layers + the output layer
type Network struct {
	Inputs []float64
	Layers []*Layer
}

// Creates a feed forward neural network with all inputs, hidden layers,
// and output layer nodes.
func NewNetwork(inputs, outputs int, hiddenLayers []int, hiddenAct, outAct ActivationFunction) *Network {
	net := &Network{
		Inputs: make([]float64, inputs),
		Layers: make([]*Layer, len(hiddenLayers)+1),
	}

	prevNodes := inputs
	for h, hidden := range hiddenLayers {
		if hidden < 1 {
			panic(fmt.Sprintf("NewNetwork ERROR number of hidden nodes %d in layer %d is below 1", hidden, h))
		}
		net.Layers[h] = createLayer(hidden, prevNodes, hiddenAct)
		prevNodes = hidden
	}

	// Add the output as the last layer
	net.Layers[len(hiddenLayers)] = createLayer(outputs, prevNodes, outAct)

	return net
}

// Trains the sample data set on the neural network updating weights for each sample
// using the stochastic gradient descent method.
func (n *Network) Train(samples, targets [][]float64, learner Learner) {
	for d, sample := range samples {
		outputs := n.Evaluate(sample)

		learner.Learn(outputs, targets[d])
	}
}

// Push the inputs into the neural network. Returning the final outputs
func (n *Network) Evaluate(inputs []float64) []float64 {
	if len(inputs) != len(n.Inputs) {
		panic(fmt.Sprintf("Network.PushInputs failed input lengths do not match %d %d", len(inputs), len(n.Inputs)))
	}

	copy(n.Inputs, inputs)

	outputs := n.Inputs
	for i := 0; i < len(n.Layers); i++ {
		outputs = propagateInputs(outputs, n.Layers[i])
	}

	return outputs
}

// Initializes weights in the network for each layer in the network
// Hidden layers are assigned weights randomly bound by high and low values.
// Output node's weights are set to 1.
func (n *Network) RandomizeWeights(low, high float64, randSrc rand.Source) {
	rnd := rand.New(randSrc)

	for _, layer := range n.Layers {
		randomizeWeights(layer.Nodes, low, high, rnd)
	}
}

// Adds inputs to a layer, computes each node of that layer,
// SKips the first input of the node because it is the bias.
func propagateInputs(inputs []float64, layer *Layer) []float64 {
	outputs := make([]float64, len(layer.Nodes))
	layer.Inputs[0] = 1 // x_j0 always 1
	copy(layer.Inputs[1:], inputs)
	for i := 0; i < len(layer.Nodes); i++ {
		outputs[i] = layer.Nodes[i].Compute()
	}
	return outputs
}

// Creates a new neural layer, with the specified number of nodes and inputs
func createLayer(nodes, inputs int, act ActivationFunction) *Layer {
	// Add room for the x_j0 and w_0 bias
	inputs++

	layer := &Layer{
		Inputs: make([]float64, inputs),
		Nodes:  make([]*Node, nodes),
	}
	for i := 0; i < nodes; i++ {
		layer.Nodes[i] = &Node{
			Inputs:     layer.Inputs,
			Weights:    make([]float64, inputs),
			Activation: act,
		}
	}

	return layer
}

// Initializes the weights to random values bounded by the high and low range.
func randomizeWeights(nodes []*Node, low, high float64, rnd *rand.Rand) {
	rng := high - low
	for _, node := range nodes {
		for i := 0; i < len(node.Weights); i++ {
			// Convert [0,1) range to low,high range for weights
			node.Weights[i] = ((rnd.Float64()) * rng) - math.Abs(low)
		}
	}
}
