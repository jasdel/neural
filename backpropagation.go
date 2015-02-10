package neural

// Implementation of the backpropagation algorithm to build a neural network learner
// uses the stochastic gradient descent algorithm to adjust weights based on error.
// Implements the Network Learner interface.
type BackPropagation struct {
	learnRate  float64
	momentum   float64
	network    *Network
	deltas     [][]float64
	prevDeltas [][]float64
}

func NewBackPropagation(network *Network, learnRate, momentum float64) *BackPropagation {
	b := &BackPropagation{
		learnRate: learnRate,
		momentum:  momentum,
		network:   network,

		// Add room for
		deltas:     make([][]float64, len(network.Layers)),
		prevDeltas: make([][]float64, len(network.Layers)),
	}

	for i, layer := range network.Layers {
		b.prevDeltas[i] = make([]float64, len(layer.Inputs))
		b.deltas[i] = make([]float64, len(layer.Inputs))

		// if i+1 == len(network.Layers) {
		// 	b.prevDeltas[i+1] = make([]float64, len(layer.Nodes))
		// 	b.deltas[i+1] = make([]float64, len(layer.Nodes))
		// }
	}

	return b
}

// Provides a way to learn the neural network using the backpropagation
// learning algorithm.
func (b *BackPropagation) Learn(outputs, targets []float64) {
	last := len(b.network.Layers) - 1

	// Calculate the output layer's deltas
	for k, output := range outputs {
		// for each output k compute the delta for target t
		// $$\delta_k \leftarrow o_k (1 - o_k) (t_k - o_k)$$
		b.deltas[last][k] = output * (1 - output) * (targets[k] - output)
	}

	// Compute the deltas for each layer.
	// Calculate delta for the nodes feeding this node. No need to process
	// the first layer in the network because the nodes feeding it are
	// just inputs
	for h := last; h > 0; h-- {
		computeDeltas(b.network.Layers[h], b.deltas[h-1], b.deltas[h])
	}

	// Update the layer's weights using the deltas calculated for each node
	for h := last; h >= 0; h-- {
		updateWeights(b.network.Layers[h], b.deltas[h], b.prevDeltas[h], b.learnRate, b.momentum)
	}

	// Swamp layers so next iteration the current layer will be the previous
	// and the current previous can be reused.
	b.prevDeltas, b.deltas = b.deltas, b.prevDeltas
}

// Computes the delta for the previous layer's node in the network given each input
// weight. Along with the deltas computed in the current layer.
func computeDeltas(layer *Layer, deltasPrev, deltas []float64) {
	// For each input h compute $$o_h (1 - o_h) \sum_{k \in nodes} w_{kh} \delta_{k}$$
	// Used instead so we can compute the deltas on the node with the inputs instead of
	// the node the inputs were outputs of.
	// Same as: for each node h compute $$o_h (1 - o_h) \sum_{k \in outputs} w_{kh} \delta_{k}$$
	for h, input := range layer.Inputs {
		sum := 0.0
		for k, node := range layer.Nodes {
			sum += node.Weights[h] * deltas[k]
		}
		deltasPrev[h] = input * (1 - input) * sum
	}
}

// Updates the weights of each layer based on the deltas already computed.
func updateWeights(layer *Layer, deltas, prevDeltas []float64, learnRate, momentum float64) {
	// For each input i update each of its weights for node j.
	// $$w_{ij}(n) \leftarrow w_{ij} + \eta \delta_j x_{ij} + \alpha w_{ij}(n-1)$$
	for i, input := range layer.Inputs {
		for j, node := range layer.Nodes {
			node.Weights[i] += learnRate*deltas[j]*input + momentum*prevDeltas[j]
		}
	}
}
