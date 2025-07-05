"""
Main file for the implementation of Sparse Cosine Optimized Policy Evolution (SCOPE)
"""

import numpy as np
from scipy.fftpack import dct


class SCOPE:
    """
    This class implements the SCOPE policy that compresses input frames using a 2-D Discrete 
    Cosine Transform (DCT), retains the top-left kxk block, sparsifies it by zeroing 
    the lowest p-th percentile coefficients, and maps it to the action space via two
    learnable linear projections and a bias.

    The policy is optimized through evolutionary strategies, where the chromosome
    encodes the weights and biases of the linear projections.
    """

    def __init__(self,
                 chromosome: list,
                 k: int,
                 p: int,
                 output_size: int):
        """Create a new SCOPE policy instance."""
        self.k = k
        self.p = p
        self.output_size = output_size
        self._process_chromosome(chromosome)

    def _process_chromosome(self, chromosome: list):
        """Split chromosome into weight and bias tensors"""
        w1_len = self.k                     # (1, k)
        w2_len = self.k * self.output_size  # (k, output_size)

        # Split the chromosome into weights and bias
        self.weights_1 = np.asarray(chromosome[:w1_len]).reshape(1, self.k)
        self.weights_2 = (np.asarray(chromosome[w1_len : w1_len + w2_len]).reshape(self.k, self.output_size))
        self.bias = (np.asarray(chromosome[w1_len + w2_len :]).reshape(1, self.output_size))

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Forward pass for the SCOPE policy"""

        # Applying the 2-D DCT to the input
        dct_rows = dct(frame.T, norm="ortho")
        dct_full = dct(dct_rows.T, norm="ortho")

        # Retain the top kxk block
        m_prime = dct_full[: self.k, : self.k].copy()

        # Sparsification step
        threshold = np.percentile(np.abs(m_prime), self.p)
        m_prime[np.abs(m_prime) < threshold] = 0.0

        # Applying the linear layers and the bias
        logits = self.weights_1 @ m_prime @ self.weights_2 + self.bias
        return logits.flatten()


def compute_chromosome_size(k: int, output_size: int) -> int:
    """Return expected length of chromosome for the current SCOPE policy"""
    return k + k * output_size + output_size
