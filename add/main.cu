#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <stdio.h>

template<class Tdata>
static __device__ void add(
    Tdata *__restrict__ c,
    int const *c_s,
    Tdata const *__restrict__ a,
    int const *a_s,
    Tdata const *__restrict__ b,
    int const *b_s,
    int const count,
    int const *i_s,
    int const i_s_size) {
    // 使用一维grid
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    // 定义c,a,b的偏移
    int c_offset = 0, a_offset = 0, b_offset = 0;

    for (size_t tmp_i = 0; tmp_i < i_s_size; ++i) {
        int k = i / i_s[tmp_i];
        c += k * c_s[tmp_i];
        a += k * a_s[tmp_i];
        b += k * b_s[tmp_i];
        i %= i_s[tmp_i];
    }
    *c = *a + *b;
}
extern "C" __global__ void  add_f16(
    half *__restrict__ c,
    int const *c_s,
    half  const *__restrict__ a,
    int const *a_s,
    half const *__restrict__ b,
    int const *b_s,
    int const count,
    int const *i_s,
    int const i_s_size) {
    add(c, c_s, a, a_s, b, b_s, count, i_s, i_s_size);
}

int main() {
    CUdevice device;
    CUcontext context;
    CUdeviceptr d_A, d_B, d_C;

    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate_v2(&context, 0, device);

    // 分配内存

    int n = 1;// 矩阵B的列数
    int c = 1;
    int h = 1024;
    int w = 1024;
    int l = n * c * h * w;


    // 分配主机内存
    half *h_A = new half[l];
    half *h_B = new half[l];
    half *h_C = new half[l];

    // 初始化矩阵A和B
    for (int i = 0; i < l; ++i) {
        h_A[i] = __float2half_rn(0.1f);
    }
    for (int i = 0; i < l; ++i) {
        h_B[i] = __float2half_rn(0.1f);
    }
    for (int i = 0; i < l; ++i) {
        h_C[i] = __float2half_rn(0.0f);
    }
    // 分配设备内存
    CUresult err = cuMemAlloc_v2(&d_A, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemAlloc_v2 d_A failed: " << std::endl;
        cuMemFree_v2(d_A);
        return 1;
    }
    err = cuMemAlloc_v2(&d_B, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemAlloc_v2 d_B failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        return 1;
    }
    err = cuMemAlloc_v2(&d_C, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemAlloc_v2 d_C failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }

    // 将数据从主机内存复制到设备内存
    err = cuMemcpyHtoD_v2(d_A, h_A, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemcpyHtoD d_A failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }
    err = cuMemcpyHtoD_v2(d_B, h_B, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemcpyHtoD d_B failed: " << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }

    // 设置矩阵乘法参数
    const half alpha = 1.0f;
    const half beta = 1.0f;
    const int stride[] = {c * h * w, w * h, w, 1};
    // 假设我们有一个数组需要处理，其大小为N

    // 定义每个线程块内的线程数量，通常是32的倍数
    int BLOCK_DIM= 256;

    // 计算所需的线程块数量，并确保至少有一个线程块
    int num_block= (l + BLOCK_DIM - 1) / BLOCK_DIM;

 
    add_f16<<<num_block, BLOCK_DIM>>>(
        reinterpret_cast<half *>(d_C),
        stride,
        reinterpret_cast<half *>(d_A),
        stride,
        reinterpret_cast<half *>(d_B),
        stride,
        l,
        stride,
        sizeof(shape) / sizeof(shape[0]));


    cuCtxSynchronize();


    // 将结果从设备内存复制回主机内存
    err = cuMemcpyDtoH_v2(h_C, d_C, l * sizeof(half));
    if (err != CUDA_SUCCESS) {
        std::cerr << "cuMemcpyDtoH_v2 d_C failed: " << err << std::endl;
        cuMemFree_v2(d_A);
        cuMemFree_v2(d_B);
        cuMemFree_v2(d_C);
        return 1;
    }


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
    // 然后销毁上下文
    CUresult result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        return 1;
    }

    return 0;
}