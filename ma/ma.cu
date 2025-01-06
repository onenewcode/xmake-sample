#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>

// Helper function to perform and time HtoD memory copy
void time_memcpyHtoD(CUdeviceptr d_A, float* h_A, size_t bytes) {
    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);

    // Record start event
    cuEventRecord(start, 0);

    // Copy from host to device
    cuMemcpyHtoD_v2(d_A, h_A, bytes);

    // Record stop event
    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cuEventElapsedTime(&milliseconds, start, stop);
    printf("Time for HtoD transfer: %f ms\n", milliseconds);

    // Cleanup events
    cuEventDestroy(start);
    cuEventDestroy(stop);
}

// Helper function to perform and time DtoH memory copy
void time_memcpyDtoH(float* h_A, CUdeviceptr d_A, size_t bytes) {
    CUevent start, stop;
    cuEventCreate(&start, 0);
    cuEventCreate(&stop, 0);

    // Record start event
    cuEventRecord(start, 0);

    // Copy from device to host
    cuMemcpyDtoH_v2(h_A, d_A, bytes);

    // Record stop event
    cuEventRecord(stop, 0);
    cuEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cuEventElapsedTime(&milliseconds, start, stop);
    printf("Time for DtoH transfer: %f ms\n", milliseconds);

    // Cleanup events
    cuEventDestroy(start);
    cuEventDestroy(stop);
}

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdeviceptr d_A;

    // Initialize the CUDA driver API
    cuInit(0);

    // Choose which GPU to use (if any)
    cuDeviceGet(&device, 0);

    // Create a CUDA context for the chosen device
    cuCtxCreate(&context, 0, device);

    size_t N = 2048 * 2048 * 32; // Example size, can be changed as needed
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);

    // Allocate device memory
    cuMemAlloc(&d_A, bytes);

    // Perform and time the HtoD memory copy
    time_memcpyHtoD(d_A, h_A, bytes);

    // Perform and time the DtoH memory copy
    time_memcpyDtoH(h_A, d_A, bytes);

    // Cleanup
    cuMemFree(d_A);
    free(h_A);
    cuCtxDestroy(context);

    return 0;
}