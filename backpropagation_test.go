package neural

import (
	"log"
	"math/rand"
	"testing"
)

func TestBPComputeDeltas(t *testing.T) {

}

func TestBPUpdateWeights(t *testing.T) {

}

func TestBPLearn(t *testing.T) {
	net := NewNetwork(5, 2, []int{3, 4, 3}, SigmoidActivation, SigmoidActivation)

	samples := [][]float64{
		[]float64{0, 2, 2, 1, 2}, // []float64{0, 2, 3, 2, 2}, []float64{0, 2, 1, 3, 2},
	}

	targets := [][]float64{
		[]float64{1, 2}, []float64{1, 1}, []float64{1, 3},
	}

	net.RandomizeWeights(-0.05, 0.05, rand.NewSource(1))
	bp := NewBackPropagation(net, 0.3, 0.2)

	for i, layer := range net.Layers {
		for j, node := range layer.Nodes {
			log.Println(i, j, node.Weights)
		}
	}

	net.Train(samples, targets, bp)

	for i, layer := range net.Layers {
		for j, node := range layer.Nodes {
			log.Println(i, j, node.Weights)
		}
	}
}

func BenchmarkBPTrain(b *testing.B) {
	inputs := 5
	outputs := 2
	count := 2000
	net := NewNetwork(inputs, outputs, []int{3, 4, 3}, SigmoidActivation, SigmoidActivation)

	samples := make([][]float64, count)
	for i := 0; i < len(samples); i++ {
		samples[i] = make([]float64, inputs)
		for j := 0; j < len(samples[i]); j++ {
			samples[i][j] = rand.Float64()
		}
	}
	targets := make([][]float64, count)
	for i := 0; i < len(targets); i++ {
		targets[i] = make([]float64, outputs)
		for j := 0; j < len(targets[i]); j++ {
			targets[i][j] = rand.Float64()
		}
	}

	net.RandomizeWeights(-0.05, 0.05, rand.NewSource(1))
	bp := NewBackPropagation(net, 0.3, 0.2)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		net.Train(samples, targets, bp)
	}
}
