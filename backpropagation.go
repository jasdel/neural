package neural

import (
	"math"
	"math/rand"
)

type BackPropagation struct {
	Network    *FeedForward
	prevDeltas [][]float64
}

// Initializes weights in the network for each layer in the network
// Hidden layers are assigned weights randomly bound by high and low values.
// Output node's weights are set to 1.
func (b *BackPropagation) InitWeights(low, high float64, randSrc rand.Source) {
	rnd := rand.New(randSrc)

	for _, layer := range b.Network.HiddenLayers {
		RandInitWeights(layer.Nodes, low, high, rnd)
	}
	RandInitWeights(b.Network.Outputs.Nodes, low, high, rnd)
	// for _, node := range b.Network.Outputs.Nodes {
	// 	// Output nodes just sum without weights
	// 	// Bias weight w[0] is skipped since not needed.
	// 	// for i := 1; i < len(node.Weights); i++ {
	// 	// 	node.Weights[i] = 1
	// 	// }
	// }
}

func (b *BackPropagation) Train(samples, targets [][]float64, learnRate, momentum float64) {
	for d, sample := range samples {
		outputs := b.Network.Forward(sample)

		last := len(b.Network.HiddenLayers)
		deltas := make([][]float64, len(b.Network.HiddenLayers)+1)
		deltas[last] = make([]float64, len(outputs))

		// Calculate the output deltas
		for k, output := range outputs {
			deltas[last][k] = output * (1 - output) * (targets[d][k] - output)
		}
		deltas[last-1] = computeDeltas(b.Network.Outputs, deltas[last])

		// Calculate the hidden deltas
		layerFwd := b.Network.Outputs
		for h := last - 1; h >= 0; h-- {
			layer := b.Network.HiddenLayers[h]
			deltas[h] = hiddenBackProp(layer, layerFwd, deltas[h+1])
			layerFwd = layer
		}

		// Update the the deltas
		layerFwd = b.Network.Outputs
		for h := last - 1; h >= 0; h-- {
			layer := b.Network.HiddenLayers[h]
			updateBackPropWeights(layer, layerFwd, deltas[h], b.prevDeltas[h+1], learnRate, momentum)
			layerFwd = layer
		}

		b.prevDeltas = deltas
	}
}

// Computes the delta for the current node's weights given each input
// and the deltas computed in the forward layer
func computeDeltas(layer *Layer, deltasFwd []float64) []float64 {
	deltasBck := make([]float64, len(layer.Inputs))

	// For each input h compute $$o_h (1 - o_h) \sum_{k \in nodes} w_{kh} \delta_{kh}$$
	// Used instead so we can calculate the deltas on the node with the inputs instead of
	// the node the inputs were outputs of.
	// Same as: for each node h compute $$o_h (1 - o_h) \sum_{k \in outputs} w_{kh} \delta_{kh}$$
	for h, input := range layer.Inputs {
		sum := 0.0
		for k, node := range layer.Nodes {
			sum += node.Weights[h] * deltasFwd[k]
		}
		deltasBck[h] = input * (1 - input) * deltasBck[h]
	}
}

// Updates the weights of each layer based on the deltas already computed.
// For each node i update each of its weights j.
// $$w_{ij}(n) \leftarrow w_{ij} + \eta \delta_j x_{ij} + \alpha w_{ij}(n-1)$$
func updateBackPropWeights(layer, layerFwd *Layer, deltas, prevDeltas []float64, learnRate, momentum float64) {
	for i, node := range layer.Nodes {
		for j := 0; j < len(layerFwd.Nodes[i].Weights); j++ {
			layerFwd.Nodes[i].Weights[j] += learnRate*deltas[j]*layerFwd.Nodes[i].Inputs[j] + momentum*prevDeltas[j]
		}
	}
}

// Initializes the weights to random values bounded by the high and low range.
func RandInitWeights(nodes []*Node, low, high float64, rnd *rand.Rand) {
	rng := high - low
	for _, node := range nodes {
		for i := 0; i < len(node.Weights); i++ {
			// Convert [0,1) range to low,high range for weights
			node.Weights[i] = ((rnd.Float64()) * rng) - math.Abs(low)
		}
	}
}
