package neural

import (
	"math"
)

type ActivationFunction func(x, w []float64, t float64) float64

func SigmoidActivation(x, w []float64, t float64) float64 {
	net := SumInputs(x, w, t)

	// o = sigma(net) = 1 / (1 + e^(-net))
	return 1 / (1 + math.Pow(math.E, -net))
}

func PerceptronActivation(x, w []float64, t float64) float64 {
	if sum := SumInputs(x, w, t); sum >= t {
		return 1
	}
	return 0
}

// Activation function which does nothing more than the summation
// of x * w, t is ignored, and only present so SumInputs matches
// the ActivationFunction type
func SumInputs(x, w []float64, t float64) float64 {
	var sum float64 = 0
	for i := 0; i < len(x); i++ {
		sum += x[i] * w[i]
	}
	return sum
}

type Node struct {
	Inputs     []float64
	Weights    []float64
	Threshold  float64
	Output     float64
	Activation ActivationFunction
}

func (n *Node) Compute() float64 {
	n.Output = n.Activation(n.Inputs, n.Weights, n.Threshold)
	return n.Output
}
