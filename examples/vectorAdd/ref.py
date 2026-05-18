"""Reference for: solve(const float* A, const float* B, float* C, int N)"""
import torch

def reference(*, A, B, C, N, **kwargs):
    C[:N] = A[:N] + B[:N]

atol = 1e-4
rtol = 1e-3
