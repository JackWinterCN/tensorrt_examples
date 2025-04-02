__global__ void average_kernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = (a[idx] + b[idx]) / 2;
  }
}

void average_op(const float *a, const float *b, float *c, int n) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  average_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
}