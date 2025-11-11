import torch
import torch.nn.functional as F

def gcmi(
    targets: torch.Tensor,
    queries: torch.Tensor,
    lambda_p: float = 0.5,
    similarity: str = "cosine",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute Graph Cut Mutual Information (GCMI) between a target set A (targets) and
    a query set Q (queries) using a cross-similarity matrix S and the formula:
        GCMI(A,Q) = 2 * lambda_p * sum_{i in A} sum_{j in Q} S[i, j]

    Args:
        targets: Tensor of shape (|A|, d) representing target set features.
        queries: Tensor of shape (|Q|, d) representing query set features.
        lambda_p: Scalar lambda controlling scale (defaults to 0.5).
        similarity: "cosine" or "dot" to define S.
        eps: Small constant for numerical stability in cosine normalization.

    Returns:
        Scalar torch.Tensor containing the GCMI value.
    """
    if targets.ndim != 2 or queries.ndim != 2:
        raise ValueError("targets and queries must be 2D tensors of shape (n, d) and (m, d)")
    if targets.size(1) != queries.size(1):
        raise ValueError("Feature dimension d must match between targets and queries")

    if similarity == "cosine":
        # Normalize rows to unit length, then S = T @ Q^T
        t = F.normalize(targets, p=2, dim=1, eps=eps)
        q = F.normalize(queries, p=2, dim=1, eps=eps)
        S = t @ q.t()
    elif similarity == "dot":
        S = targets @ queries.t()
    else:
        raise ValueError("similarity must be 'cosine' or 'dot'")

    # GCMI(A,Q) = 2 * lambda * sum(S_ij)
    val = 2.0 * lambda_p * S.sum()
    return val
