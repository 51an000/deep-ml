import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    predicted_probs=np.clip(predicted_probs, epsilon, 1-epsilon)
    predicted_probs=np.log(predicted_probs)
    loss=-np.sum(predicted_probs*true_labels,axis=1)
    return np.mean(loss)