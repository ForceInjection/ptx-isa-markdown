// Example: vector addition C = A + B.
// Follows the benchmark.py interface: solve(const float* A, const float* B, float* C, int N).
// Compare with standalone version: cuda-samples/cpp/0_Introduction/vectorAdd/vectorAdd.cu

extern "C" __global__ void solve(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
