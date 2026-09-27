import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples,n_features=X.shape
    w=np.zeros(n_features)
    b=0.0
     # Gradient descent
    for _ in range(steps):
         # Linear combination
        z = X @ w + b
        # Predicted probabilities
        y_pred = _sigmoid(z)
        dw = (1 / n_samples) * (X.T @ (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
         # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b
    # Write code here
    pass