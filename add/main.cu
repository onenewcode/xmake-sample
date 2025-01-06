#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <random>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include <stdio.h>

template<class Tdata>
static __device__ void gelu(
    Tdata *__restrict__ a,
    int const * a_s,
    Tdata const *__restrict__ b,
    int const * b_s,
    Tdata  const *__restrict__ c,
    int const * c_s,
    int const conut,
    int const * i_s,
    int const i_s_size){
    // 使用一维grid
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    // 定义c,a,b的偏移
    int c_offset = 0, a_offset = 0, b_offset = 0;

     for (size_t tmp_i = 0; tmp_i < i_s_size; ++i) {
        int k =  i/ i_s[tmp_i];
        c += k * c_s[tmp_i];
        a += k * a_s[tmp_i];
        b += k * b_s[tmp_i];
        i %= i_s[tmp_i];
    }
   *c=*a+*b;
}



int main() {
     const int N = 100;
    size_t bytes = N * sizeof(float);

    // 在主机端分配内存并初始化随机数据
    float* h_input = new float[N];
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < N; ++i) {
        h_input[i] = distribution(generator);
    }
 h_input[0]=0.036606249235670324f;
    // 打印原始数据
    std::cout << "Original Data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_input[i] << ",";
    }
    std::cout << std::endl;

    // 分配设备端内存
    half* d_data;
    cudaMalloc(&d_data, N * sizeof(half));

    // 将float转换为half并拷贝到设备端
    half* h_half_input = new half[N];
    for (int i = 0; i < N; ++i) {
        h_half_input[i] = __float2half(h_input[i]);
    }
    cudaMemcpy(d_data, h_half_input, N * sizeof(half), cudaMemcpyHostToDevice);

    // 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaDeviceSynchronize();

    // 拷贝结果回主机端
    half* h_half_output = new half[N];
    cudaMemcpy(h_half_output, d_data, N * sizeof(half), cudaMemcpyDeviceToHost);

    // 将half转换为float并打印输出结果
    std::cout << "Processed Data:" << std::endl;
    for (int i = 0; i < N; ++i) {
        float output = __half2float(h_half_output[i]);
        std::cout << output << " ";
    }
    std::cout << std::endl;

    // 清理
    delete[] h_input;
    delete[] h_half_input;
    delete[] h_half_output;
    cudaFree(d_data);

    return 0;
}