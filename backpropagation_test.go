package neural

import (
	"math/rand"
	"testing"
)

func TestBPInitWeights(t *testing.T) {
	net := NewFeedForward(2, 1, []int{1}, SigmoidActivation, SumInputs)

	low := -0.05
	high := 0.05
	bp := BackPropagation{Network: net}
	bp.InitWeights(low, high, rand.NewSource(1))

	for i, layer := range bp.Network.HiddenLayers {
		for j, node := range layer.Nodes {
			for k, weight := range node.Weights {
				if weight < low || weight > high {
					t.Errorf("Unexpected weight, %d,%d,%d, %f", i, j, k, weight)
				}
			}
		}
	}

	for i, node := range bp.Network.Outputs.Nodes {
		// if node.Weights[0] != 0 {
		// 	t.Errorf("Unexpected weight for output bias %d %f", i, node.Weights[0])
		// }
		// for j := 1; j < len(node.Weights); j++ {
		// 	if node.Weights[j] != 1 {
		// 		t.Errorf("Unexpected weight %d,%d %f", i, j, node.Weights[j])
		// 	}
		// }
		for j, weight := range node.Weights {
			if weight < low || weight > high {
				t.Errorf("Unexpected weight, %d,%d, %f", i, j, weight)
			}
		}
	}
}
