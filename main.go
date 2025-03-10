package main

import (
    "bytes"
    "encoding/gob"
    "flag"
    "fmt"
    "log"
    "math"
    "math/rand"
    "os"
    "path/filepath"
    "sync"
    "time"

    "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)

// -----------------------------
// Global Settings and Utilities
// -----------------------------

var useGPU bool

func init() {
    flag.BoolVar(&useGPU, "gpu", false, "Enable GPU acceleration")
    flag.Parse()
    rand.Seed(time.Now().UnixNano())
    log.SetFlags(log.LstdFlags | log.Lshortfile)
}

// -----------------------------
// Activation Functions & Derivatives (CPU)
// -----------------------------

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivativeFromActivation(a float64) float64 {
    // Given an already-sigmoid'ed value a, derivative = a*(1-a).
    return a * (1 - a)
}

func leakyReLU(x float64) float64 {
    if x < 0 {
        return 0.01 * x
    }
    return x
}

func leakyReLUDerivativeFromActivation(a float64) float64 {
    // If 'a' is negative, slope is 0.01; otherwise 1.
    if a < 0 {
        return 0.01
    }
    return 1
}

func relu(x float64) float64 {
    if x > 0 {
        return x
    }
    return 0
}

func reluDerivativeFromActivation(a float64) float64 {
    if a > 0 {
        return 1
    }
    return 0
}

func getActivationDerivative(actName string, activation float64) float64 {
    switch actName {
    case "sigmoid":
        return sigmoidDerivativeFromActivation(activation)
    case "leaky":
        return leakyReLUDerivativeFromActivation(activation)
    case "relu":
        return reluDerivativeFromActivation(activation)
    default:
        // Fall back to sigmoid with a warning
        log.Printf("[WARN] Unrecognized activation '%s'; defaulting derivative to sigmoid.\n", actName)
        return sigmoidDerivativeFromActivation(activation)
    }
}

// -----------------------------
// Loss Functions
// -----------------------------

type LossFunction interface {
    Compute(pred, target []float64) float64
    Derivative(pred, target []float64) []float64
}

type MSELoss struct{}

func (l MSELoss) Compute(pred, target []float64) float64 {
    var loss float64
    for i := range pred {
        err := target[i] - pred[i]
        loss += err * err
    }
    return loss / float64(len(pred))
}

func (l MSELoss) Derivative(pred, target []float64) []float64 {
    grad := make([]float64, len(pred))
    for i := range pred {
        grad[i] = -2 * (target[i] - pred[i]) / float64(len(pred))
    }
    return grad
}

type CrossEntropyLoss struct{}

func (l CrossEntropyLoss) Compute(pred, target []float64) float64 {
    var loss float64
    for i := range pred {
        // clamp pred to avoid log(0)
        p := math.Min(math.Max(pred[i], 1e-7), 1-1e-7)
        loss += -target[i]*math.Log(p) - (1-target[i])*math.Log(1-p)
    }
    return loss / float64(len(pred))
}

func (l CrossEntropyLoss) Derivative(pred, target []float64) []float64 {
    grad := make([]float64, len(pred))
    for i := range pred {
        grad[i] = (pred[i] - target[i]) / float64(len(pred))
    }
    return grad
}

// -----------------------------
// Optimizer Interfaces & Implementations
// -----------------------------

type Optimizer interface {
    Update(layerIdx int, weights, biases [][]float64, gradientsW, gradientsB [][]float64, learningRate float64)
    RegisterLayer(layerIdx int, weights, biases [][]float64)
}

// Adam optimizer (stateful).
// NOTE: We are not serializing mW, vW, etc. If you want to resume
// exactly where you left off with momentum, you’d also store these
// in your ModelData and reload them.
type Adam struct {
    beta1, beta2, epsilon float64
    timeStep              int
    mW                    [][][]float64
    vW                    [][][]float64
    mB                    [][][]float64
    vB                    [][][]float64
}

func NewAdam() *Adam {
    return &Adam{
        beta1:    0.9,
        beta2:    0.999,
        epsilon:  1e-8,
        timeStep: 0,
        mW:       make([][][]float64, 0),
        vW:       make([][][]float64, 0),
        mB:       make([][][]float64, 0),
        vB:       make([][][]float64, 0),
    }
}

func (a *Adam) RegisterLayer(layerIdx int, weights, biases [][]float64) {
    layerMW := make([][]float64, len(weights))
    layerVW := make([][]float64, len(weights))
    for i, w := range weights {
        layerMW[i] = make([]float64, len(w))
        layerVW[i] = make([]float64, len(w))
    }
    a.mW = append(a.mW, layerMW)
    a.vW = append(a.vW, layerVW)

    layerMB := make([][]float64, len(biases))
    layerVB := make([][]float64, len(biases))
    for i := range biases {
        layerMB[i] = []float64{0}
        layerVB[i] = []float64{0}
    }
    a.mB = append(a.mB, layerMB)
    a.vB = append(a.vB, layerVB)
}

func (a *Adam) Update(layerIdx int, weights, biases [][]float64, gradientsW, gradientsB [][]float64, learningRate float64) {
    a.timeStep++
    for i := range weights {
        for j := range weights[i] {
            // Update moment estimates
            a.mW[layerIdx][i][j] = a.beta1*a.mW[layerIdx][i][j] + (1-a.beta1)*gradientsW[i][j]
            a.vW[layerIdx][i][j] = a.beta2*a.vW[layerIdx][i][j] + (1-a.beta2)*gradientsW[i][j]*gradientsW[i][j]
            // Correct bias
            mHat := a.mW[layerIdx][i][j] / (1 - math.Pow(a.beta1, float64(a.timeStep)))
            vHat := a.vW[layerIdx][i][j] / (1 - math.Pow(a.beta2, float64(a.timeStep)))
            // Apply update
            weights[i][j] -= learningRate * mHat / (math.Sqrt(vHat) + a.epsilon)
        }

        // Update bias
        a.mB[layerIdx][i][0] = a.beta1*a.mB[layerIdx][i][0] + (1-a.beta1)*gradientsB[i][0]
        a.vB[layerIdx][i][0] = a.beta2*a.vB[layerIdx][i][0] + (1-a.beta2)*gradientsB[i][0]*gradientsB[i][0]

        mHatB := a.mB[layerIdx][i][0] / (1 - math.Pow(a.beta1, float64(a.timeStep)))
        vHatB := a.vB[layerIdx][i][0] / (1 - math.Pow(a.beta2, float64(a.timeStep)))
        biases[i][0] -= learningRate * mHatB / (math.Sqrt(vHatB) + a.epsilon)
    }
}

// A basic SGD optimizer (stateless).
type SGD struct{}

func NewSGD() *SGD {
    return &SGD{}
}

func (s *SGD) RegisterLayer(layerIdx int, weights, biases [][]float64) {
    // No state needed for basic SGD.
}

func (s *SGD) Update(layerIdx int, weights, biases [][]float64, gradientsW, gradientsB [][]float64, learningRate float64) {
    for i := range weights {
        for j := range weights[i] {
            weights[i][j] -= learningRate * gradientsW[i][j]
        }
        biases[i][0] -= learningRate * gradientsB[i][0]
    }
}

// -----------------------------
// Layer & Model Definitions
// -----------------------------

type Layer struct {
    Weights        [][]float64
    Biases         [][]float64
    ActivationName string // "sigmoid", "leaky", "relu"
}

type NeuralNetwork struct {
    Layers       []*Layer
    LearningRate float64
    Optimizer    Optimizer    `gob:"-"`
    Loss         LossFunction `gob:"-"`
}

func NewNeuralNetwork(learningRate float64, optimizer Optimizer, loss LossFunction) *NeuralNetwork {
    return &NeuralNetwork{
        LearningRate: learningRate,
        Optimizer:    optimizer,
        Loss:         loss,
        Layers:       []*Layer{},
    }
}

func (nn *NeuralNetwork) AddLayer(inputSize, outputSize int, activationName string) {
    layer := &Layer{
        Weights:        make([][]float64, outputSize),
        Biases:         make([][]float64, outputSize),
        ActivationName: activationName,
    }
    for i := 0; i < outputSize; i++ {
        layer.Weights[i] = make([]float64, inputSize)
        for j := 0; j < inputSize; j++ {
            // random init
            layer.Weights[i][j] = rand.Float64()*2 - 1
        }
        layer.Biases[i] = []float64{rand.Float64()*2 - 1}
    }

    // If the optimizer needs to register layer shapes, do so here.
    if reg, ok := nn.Optimizer.(interface {
        RegisterLayer(int, [][]float64, [][]float64)
    }); ok {
        reg.RegisterLayer(len(nn.Layers), layer.Weights, layer.Biases)
    }
    nn.Layers = append(nn.Layers, layer)
}

func (nn *NeuralNetwork) Compile() {
    // In a production environment, you might build a Gorgonia graph
    // that covers forward *and* backward passes, letting Gorgonia handle auto-differentiation.
    // For now, we keep the manual gradient code below.
}

// -----------------------------
// Forward Pass
// -----------------------------

func (nn *NeuralNetwork) Forward(input []float64) []float64 {
    // If requested, run the forward pass in Gorgonia on GPU, but note
    // this code does not currently do GPU-based backprop with Gorgonia.
    if useGPU {
        return nn.forwardGorgonia(input)
    }
    return nn.forwardCPU(input)
}

func (nn *NeuralNetwork) forwardCPU(input []float64) []float64 {
    output := input
    for _, layer := range nn.Layers {
        newOutput := make([]float64, len(layer.Weights))
        for i := range layer.Weights {
            sum := layer.Biases[i][0]
            for j, weight := range layer.Weights[i] {
                sum += weight * output[j]
            }
            switch layer.ActivationName {
            case "leaky":
                newOutput[i] = leakyReLU(sum)
            case "relu":
                newOutput[i] = relu(sum)
            case "sigmoid":
                newOutput[i] = sigmoid(sum)
            default:
                log.Printf("[WARN] Unrecognized activation '%s'; defaulting to sigmoid.\n", layer.ActivationName)
                newOutput[i] = sigmoid(sum)
            }
        }
        output = newOutput
    }
    return output
}

// forwardGorgonia builds a small dynamic graph each time for forward-only.
// In a real GPU-based training scenario, you'd keep a persistent graph for
// forward+backprop. This is purely to illustrate Gorgonia usage.
func (nn *NeuralNetwork) forwardGorgonia(input []float64) []float64 {
    g := gorgonia.NewGraph()

    xT := tensor.New(tensor.WithBacking(input), tensor.WithShape(len(input), 1))
    xNode := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(len(input), 1),
        gorgonia.WithName("x"), gorgonia.WithValue(xT))

    node := xNode
    for idx, layer := range nn.Layers {
        rows := len(layer.Weights)
        cols := len(layer.Weights[0])

        // Flatten Weights
        flatW := make([]float64, 0, rows*cols)
        for i := 0; i < rows; i++ {
            flatW = append(flatW, layer.Weights[i]...)
        }

        wT := tensor.New(tensor.WithBacking(flatW), tensor.WithShape(rows, cols))
        wNode := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(rows, cols),
            gorgonia.WithName(fmt.Sprintf("w_%d", idx)), gorgonia.WithValue(wT))

        // Flatten Biases
        flatB := make([]float64, rows)
        for i := 0; i < rows; i++ {
            flatB[i] = layer.Biases[i][0]
        }
        bT := tensor.New(tensor.WithBacking(flatB), tensor.WithShape(rows, 1))
        bNode := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(rows, 1),
            gorgonia.WithName(fmt.Sprintf("b_%d", idx)), gorgonia.WithValue(bT))

        mul, err := gorgonia.Mul(wNode, node)
        if err != nil {
            panic(err)
        }
        sum, err := gorgonia.Add(mul, bNode)
        if err != nil {
            panic(err)
        }

        var activated *gorgonia.Node
        switch layer.ActivationName {
        case "leaky":
            rectified, err := gorgonia.Rectify(sum)
            if err != nil {
                panic(err)
            }
            diff, err := gorgonia.Sub(sum, rectified)
            if err != nil {
                panic(err)
            }
            scaledDiff, err := gorgonia.Mul(diff, gorgonia.NewConstant(0.01))
            if err != nil {
                panic(err)
            }
            activated, err = gorgonia.Add(rectified, scaledDiff)
            if err != nil {
                panic(err)
            }
        case "relu":
            activated, err = gorgonia.Rectify(sum)
            if err != nil {
                panic(err)
            }
        case "sigmoid":
            activated, err = gorgonia.Sigmoid(sum)
            if err != nil {
                panic(err)
            }
        default:
            log.Printf("[WARN] Unrecognized activation '%s' in Gorgonia path; defaulting to sigmoid.\n", layer.ActivationName)
            activated, err = gorgonia.Sigmoid(sum)
            if err != nil {
                panic(err)
            }
        }
        node = activated
    }

    machine := gorgonia.NewTapeMachine(g)
    if err := machine.RunAll(); err != nil {
        panic(err)
    }
    outT := node.Value().(tensor.Tensor)
    return outT.Data().([]float64)
}

// -----------------------------
// Evaluation
// -----------------------------

func (nn *NeuralNetwork) Evaluate(inputs, targets [][]float64) float64 {
    var totalLoss float64
    for i, input := range inputs {
        pred := nn.Forward(input)
        totalLoss += nn.Loss.Compute(pred, targets[i])
    }
    return totalLoss / float64(len(inputs))
}

// -----------------------------
// Backprop - computeGradients
// -----------------------------

func (nn *NeuralNetwork) computeGradients(batchInput, batchTarget [][]float64) (gradW, gradB [][][]float64, avgLoss float64) {
    gradW = make([][][]float64, len(nn.Layers))
    gradB = make([][][]float64, len(nn.Layers))
    for l, layer := range nn.Layers {
        gradW[l] = make([][]float64, len(layer.Weights))
        gradB[l] = make([][]float64, len(layer.Biases))
        for i := range layer.Weights {
            gradW[l][i] = make([]float64, len(layer.Weights[i]))
            gradB[l][i] = []float64{0}
        }
    }

    var totalLoss float64

    // For each sample in the mini-batch
    for idx, input := range batchInput {
        // Forward pass (CPU version) - for grad calc
        activations := make([][]float64, len(nn.Layers)+1)
        activations[0] = input

        // Track intermediate activations
        for l, layer := range nn.Layers {
            activations[l+1] = make([]float64, len(layer.Weights))
            for i := range layer.Weights {
                sum := layer.Biases[i][0]
                for j := range layer.Weights[i] {
                    sum += layer.Weights[i][j] * activations[l][j]
                }
                switch layer.ActivationName {
                case "leaky":
                    activations[l+1][i] = leakyReLU(sum)
                case "relu":
                    activations[l+1][i] = relu(sum)
                case "sigmoid":
                    activations[l+1][i] = sigmoid(sum)
                default:
                    log.Printf("[WARN] Unrecognized activation '%s'; defaulting to sigmoid in backprop.\n", layer.ActivationName)
                    activations[l+1][i] = sigmoid(sum)
                }
            }
        }

        // Compute loss for this sample
        pred := activations[len(nn.Layers)]
        loss := nn.Loss.Compute(pred, batchTarget[idx])
        totalLoss += loss

        // Derivative of loss wrt final-layer output
        deltas := nn.Loss.Derivative(pred, batchTarget[idx])

        // For MSE, chain rule includes derivative of final activation
        if _, ok := nn.Loss.(MSELoss); ok {
            for i := range deltas {
                deltas[i] *= getActivationDerivative(nn.Layers[len(nn.Layers)-1].ActivationName, pred[i])
            }
        }

        // Backprop through layers
        for l := len(nn.Layers) - 1; l >= 0; l-- {
            layer := nn.Layers[l]
            newDeltas := make([]float64, len(layer.Weights[0]))

            // Accumulate grads for w,b
            for i := range layer.Weights {
                // For each output neuron i
                for j := range layer.Weights[i] {
                    gradW[l][i][j] += deltas[i] * activations[l][j]
                }
                gradB[l][i][0] += deltas[i]
            }
            // Prepare deltas for next layer down, if any
            if l > 0 {
                for j := range newDeltas {
                    sum := 0.0
                    for i := range layer.Weights {
                        sum += layer.Weights[i][j] * deltas[i]
                    }
                    newDeltas[j] = sum * getActivationDerivative(nn.Layers[l-1].ActivationName, activations[l][j])
                }
            }
            deltas = newDeltas
        }
    }

    return gradW, gradB, totalLoss / float64(len(batchInput))
}

// -----------------------------
// Distributed & Single-Threaded Training
// -----------------------------

// DistributedTrain performs multi-threaded gradient aggregation.
func (nn *NeuralNetwork) DistributedTrain(inputs, targets [][]float64, epochs, batchSize, numWorkers int) {
    if batchSize <= 0 {
        batchSize = len(inputs)
    }
    loader := NewDataLoader(inputs, targets, batchSize)

    for epoch := 0; epoch < epochs; epoch++ {
        var totalLoss float64

        // We'll use a wait group + channel to gather partial gradients
        var wg sync.WaitGroup
        type gradResult struct {
            gradW [][][]float64
            gradB [][][]float64
            loss  float64
        }
        // For each batch, we'll re-create a channel. This is somewhat
        // unusual but works fine for demonstration. Alternatively, you
        // can keep a single channel outside the loop, etc.
        resultsCh := make(chan gradResult, numWorkers)

        for {
            batchInputs, batchTargets, ok := loader.NextBatch()
            if !ok {
                break
            }
            partSize := (len(batchInputs) + numWorkers - 1) / numWorkers

            // Spawn workers
            for w := 0; w < numWorkers; w++ {
                start := w * partSize
                end := start + partSize
                if end > len(batchInputs) {
                    end = len(batchInputs)
                }
                if start >= end {
                    continue
                }
                wg.Add(1)

                go func(bInputs, bTargets [][]float64) {
                    defer wg.Done()
                    gW, gB, loss := nn.computeGradients(bInputs, bTargets)
                    resultsCh <- gradResult{gradW: gW, gradB: gB, loss: loss}
                }(batchInputs[start:end], batchTargets[start:end])
            }

            wg.Wait()
            close(resultsCh)

            // Aggregate partial gradients
            aggGradW := make([][][]float64, len(nn.Layers))
            aggGradB := make([][][]float64, len(nn.Layers))
            for l, layer := range nn.Layers {
                aggGradW[l] = make([][]float64, len(layer.Weights))
                aggGradB[l] = make([][]float64, len(layer.Biases))
                for i := range layer.Weights {
                    aggGradW[l][i] = make([]float64, len(layer.Weights[i]))
                    aggGradB[l][i] = []float64{0}
                }
            }

            count := 0
            for res := range resultsCh {
                count++
                totalLoss += res.loss
                for l := range aggGradW {
                    for i := range aggGradW[l] {
                        for j := range aggGradW[l][i] {
                            aggGradW[l][i][j] += res.gradW[l][i][j]
                        }
                        aggGradB[l][i][0] += res.gradB[l][i][0]
                    }
                }
            }

            if count > 0 {
                // Average the gradients
                for l := range aggGradW {
                    for i := range aggGradW[l] {
                        for j := range aggGradW[l][i] {
                            aggGradW[l][i][j] /= float64(count)
                        }
                        aggGradB[l][i][0] /= float64(count)
                    }
                }
                // Update model
                for l := range nn.Layers {
                    nn.Optimizer.Update(l,
                        nn.Layers[l].Weights,
                        nn.Layers[l].Biases,
                        aggGradW[l],
                        aggGradB[l],
                        nn.LearningRate)
                }
            }

            // Re-create channel for next batch iteration
            resultsCh = make(chan gradResult, numWorkers)
        }

        loader.Reset()
        if epoch%1000 == 0 {
            log.Printf("Distributed Epoch %d - Loss: %.6f\n",
                epoch, totalLoss/float64(len(inputs)/batchSize))
        }
    }
}

// Train is the single-threaded version. We do just ONE computeGradients call per batch.
func (nn *NeuralNetwork) Train(inputs, targets [][]float64, epochs, batchSize int) {
    if batchSize <= 0 {
        batchSize = len(inputs)
    }
    loader := NewDataLoader(inputs, targets, batchSize)

    for epoch := 0; epoch < epochs; epoch++ {
        var totalLoss float64

        for {
            batchInputs, batchTargets, ok := loader.NextBatch()
            if !ok {
                break
            }
            // Single forward/backward pass for the batch
            gradW, gradB, loss := nn.computeGradients(batchInputs, batchTargets)
            totalLoss += loss

            // Apply the gradient updates
            for l := range nn.Layers {
                nn.Optimizer.Update(l,
                    nn.Layers[l].Weights,
                    nn.Layers[l].Biases,
                    gradW[l],
                    gradB[l],
                    nn.LearningRate)
            }
        }

        loader.Reset()
        if epoch%1000 == 0 {
            log.Printf("Epoch %d - Loss: %.6f\n", epoch, totalLoss/float64(len(inputs)))
            // Checkpoint occasionally
            if epoch%5000 == 0 {
                checkpoint(nn, filepath.Join("checkpoints", fmt.Sprintf("checkpoint_epoch_%d.gob", epoch)))
            }
        }
    }
}

func (nn *NeuralNetwork) Predict(inputs [][]float64) [][]float64 {
    results := make([][]float64, len(inputs))
    for i, input := range inputs {
        results[i] = nn.Forward(input)
    }
    return results
}

// -----------------------------
// DataLoader with Shuffling
// -----------------------------

type DataLoader struct {
    Inputs    [][]float64
    Targets   [][]float64
    BatchSize int
    index     int
}

func NewDataLoader(inputs, targets [][]float64, batchSize int) *DataLoader {
    shuffledInputs, shuffledTargets := shuffleData(inputs, targets)
    return &DataLoader{
        Inputs:    shuffledInputs,
        Targets:   shuffledTargets,
        BatchSize: batchSize,
        index:     0,
    }
}

func (dl *DataLoader) NextBatch() ([][]float64, [][]float64, bool) {
    if dl.index >= len(dl.Inputs) {
        return nil, nil, false
    }
    end := dl.index + dl.BatchSize
    if end > len(dl.Inputs) {
        end = len(dl.Inputs)
    }
    batchInputs := dl.Inputs[dl.index:end]
    batchTargets := dl.Targets[dl.index:end]
    dl.index = end
    return batchInputs, batchTargets, true
}

func (dl *DataLoader) Reset() {
    dl.index = 0
    dl.Inputs, dl.Targets = shuffleData(dl.Inputs, dl.Targets)
}

func shuffleData(inputs, targets [][]float64) ([][]float64, [][]float64) {
    n := len(inputs)
    shuffledInputs := make([][]float64, n)
    shuffledTargets := make([][]float64, n)
    perm := rand.Perm(n)
    for i, idx := range perm {
        shuffledInputs[i] = inputs[idx]
        shuffledTargets[i] = targets[idx]
    }
    return shuffledInputs, shuffledTargets
}

// -----------------------------
// Checkpointing
// -----------------------------

func checkpoint(nn *NeuralNetwork, path string) {
    if err := nn.SaveModel(path); err != nil {
        log.Printf("Error checkpointing model: %v\n", err)
    } else {
        log.Printf("Model checkpoint saved to %s\n", path)
    }
}

// -----------------------------
// Model Saving/Loading via ModelData
// -----------------------------

type ModelData struct {
    LearningRate float64
    Layers       []LayerData
    // NOTE: If you need to store Adam's mW,vW,mB,vB for exact restart,
    // add them here and adjust Save/Load logic accordingly.
}

type LayerData struct {
    Weights        [][]float64
    Biases         [][]float64
    ActivationName string
}

func (nn *NeuralNetwork) SaveModel(path string) error {
    var md ModelData
    md.LearningRate = nn.LearningRate
    md.Layers = make([]LayerData, len(nn.Layers))

    for i, layer := range nn.Layers {
        md.Layers[i] = LayerData{
            Weights:        layer.Weights,
            Biases:         layer.Biases,
            ActivationName: layer.ActivationName,
        }
    }

    var buf bytes.Buffer
    enc := gob.NewEncoder(&buf)
    if err := enc.Encode(md); err != nil {
        return err
    }
    return os.WriteFile(path, buf.Bytes(), 0644)
}

func LoadModel(path string, optimizer Optimizer, loss LossFunction) (*NeuralNetwork, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    buf := bytes.NewBuffer(data)
    dec := gob.NewDecoder(buf)

    var md ModelData
    if err := dec.Decode(&md); err != nil {
        return nil, err
    }

    nn := NewNeuralNetwork(md.LearningRate, optimizer, loss)
    for _, ld := range md.Layers {
        inputSize := len(ld.Weights[0])
        outputSize := len(ld.Weights)
        nn.AddLayer(inputSize, outputSize, ld.ActivationName)
        nn.Layers[len(nn.Layers)-1].Weights = ld.Weights
        nn.Layers[len(nn.Layers)-1].Biases = ld.Biases
    }
    return nn, nil
}

// -----------------------------
// Visualization: Export DOT File for GraphViz
// -----------------------------

func VisualizeGraph(nn *NeuralNetwork, path string) error {
    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer f.Close()

    _, err = f.WriteString("digraph G {\n")
    if err != nil {
        return err
    }
    for i, layer := range nn.Layers {
        _, err = f.WriteString(fmt.Sprintf(
            "  layer%d [label=\"Layer %d: %s, neurons: %d\"];\n",
            i, i, layer.ActivationName, len(layer.Weights)))
        if err != nil {
            return err
        }
        if i > 0 {
            _, err = f.WriteString(fmt.Sprintf("  layer%d -> layer%d;\n", i-1, i))
            if err != nil {
                return err
            }
        }
    }
    _, err = f.WriteString("}\n")
    return err
}

// -----------------------------
// Main: Putting It All Together
// -----------------------------

func main() {
    // In larger codebases, you'd split these types into separate .go files.
    // Also add unit tests to compare your gradient computations with numeric checks.

    // Create checkpoint directory.
    checkpointDir := "./checkpoints"
    if err := os.MkdirAll(checkpointDir, os.ModePerm); err != nil {
        log.Fatalf("Error creating checkpoint directory: %v", err)
    }

    if useGPU {
        log.Println("Running GPU-accelerated forward propagation using Gorgonia!")
    } else {
        log.Println("Running in CPU mode.")
    }

    // Use CrossEntropyLoss for binary classification (XOR).
    nn := NewNeuralNetwork(0.1, NewAdam(), CrossEntropyLoss{})

    // Build a network for XOR.
    nn.AddLayer(2, 4, "leaky")
    nn.AddLayer(4, 4, "leaky")
    nn.AddLayer(4, 4, "leaky")
    nn.AddLayer(4, 1, "sigmoid")

    // XOR dataset
    inputs := [][]float64{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    }
    targets := [][]float64{
        {0},
        {1},
        {1},
        {0},
    }

    nn.Compile()

    log.Println("Starting training (single-threaded)...")
    nn.Train(inputs, targets, 10000, 0)

    // Checkpoint after initial training.
    checkpoint(nn, filepath.Join(checkpointDir, "model_initial.gob"))

    // Save the entire model.
    if err := nn.SaveModel("model.gob"); err != nil {
        log.Printf("Error saving model: %v\n", err)
    } else {
        log.Println("Model saved successfully.")
    }

    // Visualize the network structure
    if err := VisualizeGraph(nn, "./graph.dot"); err != nil {
        log.Printf("Error visualizing graph: %v\n", err)
    } else {
        log.Println("Graph visualization saved to graph.dot")
    }

    log.Println("Starting distributed training with 4 workers...")
    nn.DistributedTrain(inputs, targets, 10000, 0, 4)

    // Final checkpoint
    checkpoint(nn, filepath.Join(checkpointDir, "model_final.gob"))

    // Evaluate predictions
    log.Println("Predictions:")
    for _, sample := range inputs {
        output := nn.Forward(sample)
        log.Printf("Input: %v, Output: %v\n", sample, output)
    }

    // Reload the model to confirm correctness
    loadedModel, err := LoadModel("model.gob", NewAdam(), CrossEntropyLoss{})
    if err != nil {
        log.Printf("Error loading model: %v\n", err)
    } else {
        evalLoss := loadedModel.Evaluate(inputs, targets)
        log.Printf("Model loaded successfully. Evaluation loss: %.6f\n", evalLoss)
    }

    log.Println("Training Complete and model exported!")
}
