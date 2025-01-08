#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cuda.h>

int main() {
    CUdevice device;
    CUcontext context;
    CUfunction function;
    CUdeviceptr d_A,d_B, d_C;
    // Initialize the CUDA driver API
    // gpu::init
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate_v2(&context, 0, device);
    cuCtxPopCurrent_v2(NULL);


    // op new
    cublasHandle_t handle;
    cuCtxPushCurrent_v2(context);
    cublasCreate_v2(&handle);
    cuCtxPopCurrent_v2(NULL);

// apply
    cuCtxPushCurrent_v2(context);
    // 创建流
    cudaStream_t stream;
    CUresult err = cuStreamCreate(&stream,0);

    if (err != CUDA_SUCCESS) {
      std::cerr << "create stream fail" << std::endl;
    return 1;
    }
 
   // 分配内存

    int m = 2048; // 矩阵A的行数
    int n = 1024; // 矩阵B的列数
    int k = 2048; // 矩阵A的列数和矩阵B的行数
    int batch_count = 20; // 批量数量

    // 分配主机内存
    half *h_A = new half[m * k * batch_count];
    half *h_B = new half[k * n * batch_count];
    half *h_C = new half[m * n * batch_count];

    // 初始化矩阵A和B
    for (int i = 0; i < m * k * batch_count; ++i) {
        h_A[i] = __float2half_rn(0.1f);
    }
    for (int i = 0; i < k * n * batch_count; ++i) {
        h_B[i] = __float2half_rn(0.1f);
    }
      for (int i = 0; i < m * n* batch_count; ++i) {
        h_C[i] = __float2half_rn(0.1f);
    }
    // 分配设备内存
    err = cuMemAlloc_v2(&d_A, m * k * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemAlloc_v2 d_A failed: "  << std::endl;
        cuMemFree_v2(d_A);
        return 1;
    }
    err = cuMemAlloc_v2(&d_B, k * n * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemAlloc_v2 d_B failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        return 1;
    }
    err = cuMemAlloc_v2(&d_C, m * n * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemAlloc_v2 d_C failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }

    // 将数据从主机内存复制到设备内存
    err = cuMemcpyHtoD_v2(d_A, h_A, m * k * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemcpyHtoD d_A failed: "<< std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }
    err = cuMemcpyHtoD_v2(d_B, h_B, k * n * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemcpyHtoD d_B failed: "  << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }

    // 设置矩阵乘法参数
    const half alpha = 1.0f;
    const half beta = 1.0f;
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

    // cublas
    cublasStatus_t stat = cublasSetStream_v2(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS) {
         std::cerr << "set stream fail" << std::endl;
     return 1;
    }
    // 调用cublasGemmStridedBatchedEx进行矩阵乘法
    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k, &alpha,
        reinterpret_cast<const void*>(d_A), CUDA_R_16F, lda, strideA,
        reinterpret_cast<const void*>(d_B), CUDA_R_16F, ldb, strideB,
        &beta, reinterpret_cast<void*>(d_C), CUDA_R_16F, ldc, strideC,
        batch_count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
    );

    // 检查cuBLAS调用是否成功
    if (status!= CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << status << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        cublasDestroy(handle);
        return 1;
    }
    cuCtxSynchronize();
    
    // 确保所有CUDA操作已完成
    //  err1 = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess) {
    //     std::cerr << "sync stream fail" << std::endl;
    // return 1;
    // }
    // 记录矩阵乘法结束时间
    auto end_matmul = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_matmul = end_matmul - start_matmul;
    std::cout << "计算时间 " << elapsed_matmul.count() << " seconds." << std::endl;

    // 记录数据复制回主机开始时间
    auto start_copyback = std::chrono::system_clock::now();

    // 将结果从设备内存复制回主机内存
    err = cuMemcpyDtoH_v2(h_C, d_C, m * n * batch_count * sizeof(half));
    if (err!= CUDA_SUCCESS    ) {
        std::cerr << "cuMemcpyDtoH_v2 d_C failed: "<<err << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        cublasDestroy(handle);
        return 1;
    }

    // 记录数据复制回主机结束时间
    auto end_copyback = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_copyback = end_copyback - start_copyback;
    std::cout << "拷贝内存时间" << elapsed_copyback.count() << " seconds." << std::endl;

    // 输出结果矩阵C的前10个元素
    for (int i = 0; i < 10; ++i) {
        float output = __half2float(h_C[i]);
        std::cout << output << " ";
    }
    std::cout << std::endl;
    // 释放设备内存
    cuMemFree_v2(d_A);
    cuMemFree_v2(d_B);
    cuMemFree_v2(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
   cuCtxPopCurrent_v2(NULL);
    // 销毁cuBLAS句柄
    cublasDestroy(handle);

    // 销毁 CUDA 上下文
    cuCtxDestroy(context);

    return 0;
}