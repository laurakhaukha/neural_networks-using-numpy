"""
Neural network from scratch with NumPy.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Layer:
    """Base class. Every layer maps x -> y in forward and dL/dy -> dL/dx in backward."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(parameter, gradient) pairs. Empty for layers without parameters."""
        return []


class Dense(Layer):
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, init: str = "he"):
        # He init suits ReLU; Xavier suits tanh/sigmoid.
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown init '{init}'")
        self.W = rng.standard_normal((in_features, out_features)) * scale
        self.b = np.zeros((1, out_features))
        # Gradient buffers are overwritten (not accumulated) on every backward
        # pass, so there is no need for a separate zero_grad() step.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x  # cache input for backward
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW[...] = self.x.T @ grad
        self.db[...] = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Tanh(Layer):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Dropout(Layer):
    """Inverted dropout: scales at train time so inference needs no change."""

    def __init__(self, p: float, rng: np.random.Generator):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = p
        self.rng = rng
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy, fused.

    Fusing the two gives a simple gradient, (probs - one_hot) / N, and lets us
    use log-softmax for numerical stability.
    """

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.probs = np.exp(log_probs)
        self.y = y
        return float(-log_probs[np.arange(len(y)), y].mean())

    def backward(self) -> np.ndarray:
        n = len(self.y)
        grad = self.probs.copy()
        grad[np.arange(n), self.y] -= 1.0
        return grad / n


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Sequential:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        return [pg for layer in self.layers for pg in layer.params()]

    def predict(self, x):
        return self.forward(x, training=False).argmax(axis=1)

    def num_params(self) -> int:
        return sum(p.size for p, _ in self.params())

    def save(self, path: str) -> None:
        np.savez(path, *[p for p, _ in self.params()])

    def load(self, path: str) -> None:
        data = np.load(path)
        for i, (p, _) in enumerate(self.params()):
            p[...] = data[f"arr_{i}"]


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
# Weight decay is applied only to weight matrices (ndim > 1), not to biases.

class SGD:
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p) for p, _ in params]

    def step(self):
        for (p, g), v in zip(self.params, self.velocity):
            if self.weight_decay and p.ndim > 1:
                g = g + self.weight_decay * p
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for (p, g), m, v in zip(self.params, self.m, self.v):
            if self.weight_decay and p.ndim > 1:
                g = g + self.weight_decay * p
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_spiral(n_per_class: int, n_classes: int, noise: float, rng: np.random.Generator):
    """Interleaved 2-D spirals: not linearly separable, so a good NN test."""
    n = n_per_class
    X = np.zeros((n * n_classes, 2))
    y = np.zeros(n * n_classes, dtype=int)
    for c in range(n_classes):
        idx = slice(c * n, (c + 1) * n)
        radius = np.linspace(0.0, 1.0, n)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, n) + rng.standard_normal(n) * noise
        X[idx] = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
        y[idx] = c
    return X, y


def train_val_test_split(X, y, rng, val_frac=0.15, test_frac=0.15):
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * val_frac)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


def iterate_minibatches(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loss_fn, X, y):
    logits = model.forward(X, training=False)
    loss = loss_fn.forward(logits, y)
    acc = float((logits.argmax(axis=1) == y).mean())
    return loss, acc


def train(model, loss_fn, optimizer, train_data, val_data, epochs, batch_size, rng, log_every=25):
    X_train, y_train = train_data
    X_val, y_val = val_data
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            logits = model.forward(xb, training=True)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            optimizer.step()
            running += loss * len(xb)

        train_loss = running / len(X_train)
        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:4d} | train loss {train_loss:.4f} | "
                  f"val loss {val_loss:.4f} | val acc {val_acc:.3f}")
    return history


def gradient_check(model, loss_fn, X, y, rng, eps=1e-5, checks_per_param=10) -> float:
    """Compare backprop gradients with central finite differences.

    Returns the worst relative error; values below ~1e-6 mean backprop is correct.
    """
    loss_fn.forward(model.forward(X, training=False), y)
    model.backward(loss_fn.backward())

    worst = 0.0
    for p, g in model.params():
        analytic = g.copy()
        for _ in range(checks_per_param):
            i = tuple(rng.integers(0, s) for s in p.shape)
            original = p[i]
            p[i] = original + eps
            loss_plus = loss_fn.forward(model.forward(X, training=False), y)
            p[i] = original - eps
            loss_minus = loss_fn.forward(model.forward(X, training=False), y)
            p[i] = original
            numeric = (loss_plus - loss_minus) / (2 * eps)
            rel = abs(numeric - analytic[i]) / max(1e-12, abs(numeric) + abs(analytic[i]))
            worst = max(worst, rel)
    return worst


def plot_results(model, X, y, history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot (pip install matplotlib).")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Decision boundary
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid_pred = model.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax1.contourf(xx, yy, grid_pred, alpha=0.3, cmap="viridis")
    ax1.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="viridis", edgecolors="k", linewidths=0.3)
    ax1.set_title("Decision boundary")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Learning curves
    ax2.plot(history["train_loss"], label="train loss")
    ax2.plot(history["val_loss"], label="val loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross-entropy")
    ax2.set_title("Learning curves")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(in_features, hidden, n_classes, dropout, rng):
    return Sequential(
        Dense(in_features, hidden, rng),
        ReLU(),
        Dropout(dropout, rng),
        Dense(hidden, hidden, rng),
        ReLU(),
        Dropout(dropout, rng),
        Dense(hidden, n_classes, rng, init="xavier"),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Train a NumPy-only neural network on a spiral dataset.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64, help="units per hidden layer")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-2 for adam, 0.1 for sgd")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--samples", type=int, default=300, help="samples per class")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="save decision boundary + learning curves")
    ap.add_argument("--save", type=str, default=None, help="path to save weights (.npz)")
    ap.add_argument("--skip-gradcheck", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. Verify backprop on a tiny network before training anything.
    #    Smooth activations are used here on purpose: ReLU has a kink at 0, and
    #    finite differences that straddle it give misleading errors.
    if not args.skip_gradcheck:
        Xg, yg = make_spiral(10, 3, 0.2, rng)
        tiny = Sequential(
            Dense(2, 8, rng, init="xavier"), Tanh(),
            Dense(8, 8, rng, init="xavier"), Sigmoid(),
            Dense(8, 3, rng, init="xavier"),
        )
        err = gradient_check(tiny, SoftmaxCrossEntropy(), Xg, yg, rng)
        status = "OK" if err < 1e-6 else "CHECK YOUR BACKWARD PASS"
        print(f"Gradient check: max relative error = {err:.2e}  [{status}]\n")

    # 2. Data: generate, split, standardise using training statistics only.
    X, y = make_spiral(args.samples, args.classes, args.noise, rng)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y, rng)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    X_tr, X_va, X_te = (X_tr - mean) / std, (X_va - mean) / std, (X_te - mean) / std

    # 3. Model, loss, optimiser.
    model = build_model(X.shape[1], args.hidden, args.classes, args.dropout, rng)
    loss_fn = SoftmaxCrossEntropy()
    if args.optimizer == "adam":
        optimizer = Adam(model.params(), lr=args.lr or 1e-2, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.params(), lr=args.lr or 0.1, momentum=0.9, weight_decay=args.weight_decay)

    print(f"Data: train {len(X_tr)}, val {len(X_va)}, test {len(X_te)} | classes {args.classes}")
    print(f"Model: {model.num_params():,} parameters | optimiser {args.optimizer}\n")

    # 4. Train.
    start = time.perf_counter()
    history = train(model, loss_fn, optimizer, (X_tr, y_tr), (X_va, y_va),
                    args.epochs, args.batch_size, rng)
    elapsed = time.perf_counter() - start

    # 5. Final evaluation on held-out test data.
    test_loss, test_acc = evaluate(model, loss_fn, X_te, y_te)
    print(f"\nTraining took {elapsed:.1f}s")
    print(f"Test loss {test_loss:.4f} | test accuracy {test_acc:.3f}")

    if args.save:
        model.save(args.save)
        print(f"Saved weights to {args.save}")

    if args.plot:
        X_all = np.vstack([X_tr, X_va, X_te])
        y_all = np.concatenate([y_tr, y_va, y_te])
        plot_results(model, X_all, y_all, history, "results.png")


if __name__ == "__main__":
    main()
