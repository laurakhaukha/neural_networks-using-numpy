# Neural Network from Scratch (NumPy)

A fully connected neural network built using only NumPy — no PyTorch, no TensorFlow. Every forward pass, backward pass and optimiser update is written out by hand, so you can read exactly how training works.

![Decision boundary and learning curves](results.png)

## What's inside

| Component | Details |
|---|---|
| Layers | `Dense` (He / Xavier init), `Dropout` (inverted) |
| Activations | `ReLU`, `Tanh`, `Sigmoid` |
| Loss | Softmax + cross-entropy, fused and numerically stable (log-softmax) |
| Optimisers | SGD with momentum, Adam, optional L2 weight decay |
| Training | Mini-batches, train / validation / test split, feature standardisation |
| Verification | Numerical gradient check against finite differences |
| Extras | Save / load weights (`.npz`), decision-boundary plot |

## Quick start

```bash
pip install -r requirements.txt
python nn_from_scratch.py --plot
```

Example output:

```
Gradient check: max relative error = 3.39e-07  [OK]

Data: train 630, val 135, test 135 | classes 3
Model: 4,547 parameters | optimiser adam

epoch    1 | train loss 0.7788 | val loss 0.6193 | val acc 0.696
epoch   25 | train loss 0.0176 | val loss 0.0366 | val acc 0.993
...
epoch  300 | train loss 0.0100 | val loss 0.0180 | val acc 0.993

Training took 0.8s
Test loss 0.0173 | test accuracy 1.000
```

## Options

```bash
python nn_from_scratch.py --optimizer sgd --lr 0.1 --epochs 500
python nn_from_scratch.py --hidden 128 --dropout 0.2 --classes 5 --noise 0.3
python nn_from_scratch.py --save weights.npz
python nn_from_scratch.py --help
```

## How it works

Each layer implements two methods:

- `forward(x)` computes the output and caches what it needs.
- `backward(grad)` receives dL/d(output) and returns dL/d(input), storing gradients for its parameters along the way.

`Sequential` chains layers forward, then walks them in reverse for backpropagation. The fused softmax + cross-entropy loss has the simple gradient `(probs - one_hot(y)) / N`, which is where backprop starts. Optimisers then update each parameter in place using its stored gradient.

The gradient check perturbs individual weights by ±ε, measures the change in loss, and compares that numerical slope with the gradient from backprop. A relative error below 1e-6 means the backward pass is correct.

## Ideas for extending it

- Batch normalisation layer
- Learning-rate schedules and early stopping on validation loss
- Train on MNIST (flatten 28×28 images to 784 inputs)
- Regression with an MSE loss
- Convolutional layer using `im2col`

## Requirements

- Python 3.9+
- NumPy
- matplotlib (optional, only for `--plot`)
