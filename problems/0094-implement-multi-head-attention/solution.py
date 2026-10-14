import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q=X@W_q
    K=X@W_k
    V=X@W_v
    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k=Q.shape[1]
    scores=Q@K.T/np.sqrt(d_k)
    exp_scores=np.exp(scores-np.max(scores,axis=1,keepdims=True))
    weights=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    return weights@V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len,d_model=Q.shape
    assert d_model%n_heads==0
    Q_heads=np.split(Q,n_heads,axis=1)
    K_heads=np.split(K,n_heads,axis=1)
    V_heads=np.split(V,n_heads,axis=1)
    head_outputs=[]
    for i in range(n_heads):
        head_output=self_attention(Q_heads[i],K_heads[i],V_heads[i])
        head_outputs.append(head_output)
    output=np.concatenate(head_outputs, axis=1)
    return output