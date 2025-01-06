#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cuda.h>

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdeviceptr d_A,d_B, d_C;

    // Initialize the CUDA driver API
    cuInit(0);

    // Choose which GPU to use (if any)
    cuDeviceGet(&device, 0);

    // Create a CUDA context for the chosen device
    cuCtxCreate(&context, 0, device);

    int m = 2048; // 矩阵A的行数
    int n = 1024; // 矩阵B的列数
    int k = 2048; // 矩阵A的列数和矩阵B的行数
    int batch_count = 10; // 批量数量

    // 分配主机内存
    half *h_A = new half[m * k * batch_count];
    half *h_B = new half[k * n * batch_count];
    half *h_C = new half[m * n * batch_count];

    // 初始化矩阵A和B
    for (int i = 0; i < m * k * batch_count; ++i) {
        h_A[i] =__float2half_rn(1.0f) ;
    }
    for (int i = 0; i < k * n * batch_count; ++i) {
        h_B[i] = __float2half_rn(1.0f);
    }

    cuMemAlloc(&d_A, m * k * batch_count * sizeof(half));
    cuMemAlloc(&d_B, k * n * batch_count * sizeof(half));
    cuMemAlloc(&d_C, m * n * batch_count * sizeof(half));

    // 将数据从主机内存复制到设备内存
    cuMemcpyHtoD_v2(d_A, h_A, m * k * batch_count * sizeof(half));
    cuMemcpyHtoD_v2(d_B, h_B, k * n * batch_count * sizeof(half));

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 设置矩阵乘法参数
    const half alpha = 1.0f;
    const half beta = 0.0f;
    const cublasOperation_t transA = CUBLAS_OP_N;
    const cublasOperation_t transB = CUBLAS_OP_N;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    const int strideA = m * k;
    const int strideB = k * n;
    const int strideC = m * n;
    // 记录开始时间
    auto start_matmul = std::chrono::system_clock::now();
    // 调用cublasGemmStridedBatchedEx进行矩阵乘法
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k, &alpha,
        &d_A, CUDA_R_16F, lda, strideA,
        &d_B, CUDA_R_16F, ldb, strideB,
        &beta, &d_C, CUDA_R_16F, ldc, strideC,
        batch_count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
    );

    // 检查cuBLAS调用是否成功
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << status << std::endl;
        return 1;
    }



    // 确保所有CUDA操作已完成
    cuCtxSynchronize();
    // 记录矩阵乘法结束时间
    auto end_matmul = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_matmul = end_matmul - start_matmul;
    std::cout << "计算时间 " << elapsed_matmul.count() << " seconds." << std::endl;

    // 记录数据复制回主机开始时间
    auto start_copyback = std::chrono::system_clock::now();

    // 将结果从设备内存复制回主机内存
    cuMemcpyDtoH_v2(h_C, d_C, m * n * batch_count * sizeof(half));

    // 记录数据复制回主机结束时间
    auto end_copyback = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_copyback = end_copyback - start_copyback;
    std::cout << "拷贝内存时间" << elapsed_copyback.count() << " seconds." << std::endl;
    // 释放设备内存
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // 销毁cuBLAS句柄
    cublasDestroy(handle);

    return 0;
}
