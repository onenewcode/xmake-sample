/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <math.h>
// For the MACA runtime routines
#include <mc_runtime.h>
#include <mc_common.h>

/**
 * MACA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for MACA calls
  mcError_t err = mcSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = mcMalloc((void **)&d_A, size);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = mcMalloc((void **)&d_B, size);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = mcMalloc((void **)&d_C, size);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the MACA device\n");
  err = mcMemcpy(d_A, h_A, size, mcMemcpyHostToDevice);

  if (err != mcSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = mcMemcpy(d_B, h_B, size, mcMemcpyHostToDevice);

  if (err != mcSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the vectorAdd kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("MACA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = mcGetLastError();

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the MACA device to the host memory\n");
  err = mcMemcpy(h_C, d_C, size, mcMemcpyDeviceToHost);

  if (err != mcSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = mcFree(d_A);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = mcFree(d_B);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = mcFree(d_C);

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("MACA Sample Done\n");
  return 0;
}
