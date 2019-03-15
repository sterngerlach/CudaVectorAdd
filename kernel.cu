
/* kernel.cu */

#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void AddVector(
    int vecSize, const float* vecA, const float* vecB, float* vecC)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < vecSize)
        vecC[i] = vecA[i] + vecB[i];
}

int main(int argc, char** argv)
{
    const int vecSize = 16384;
    cudaError_t cudaErr = cudaError::cudaSuccess;

    std::cout << "vector addition of " << vecSize << " elements\n";

    /* ベクトル用のメモリ領域を確保 */
    float* hostVecA = new (std::nothrow) float[vecSize];

    if (hostVecA == nullptr) {
        std::cerr << "failed to allocate sufficient memory for vector A\n";
        goto Cleanup;
    }

    float* hostVecB = new (std::nothrow) float[vecSize];

    if (hostVecB == nullptr) {
        std::cerr << "failed to allocate sufficient memory for vector B\n";
        goto Cleanup;
    }

    float* hostVecC = new (std::nothrow) float[vecSize];

    if (hostVecC == nullptr) {
        std::cerr << "failed to allocate sufficient memory for vector C\n";
        goto Cleanup;
    }

    /* ベクトルAとBを初期化 */
    for (int i = 0; i < vecSize; ++i) {
        hostVecA[i] = std::rand() / static_cast<float>(RAND_MAX);
        hostVecB[i] = std::rand() / static_cast<float>(RAND_MAX);
    }

    std::cout << "vector A and B initialized\n";

    /* デバイスのベクトル用のメモリ領域を確保 */
    float* deviceVecA = nullptr;
    cudaErr = ::cudaMalloc(&deviceVecA, vecSize * sizeof(float));

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device vector A: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    float* deviceVecB = nullptr;
    cudaErr = ::cudaMalloc(&deviceVecB, vecSize * sizeof(float));

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device vector B: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    float* deviceVecC = nullptr;
    cudaErr = ::cudaMalloc(&deviceVecC, vecSize * sizeof(float));

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to allocate device vector C: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* ベクトルAとBをホストからデバイスに転送 */
    cudaErr = ::cudaMemcpy(deviceVecA, hostVecA, vecSize * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy vector A from host to device: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    cudaErr = ::cudaMemcpy(deviceVecB, hostVecB, vecSize * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy vector B from host to device: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* デバイス上でベクトルの加算を実行 */
    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (vecSize + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 dimGrid { blocksPerGrid, 1, 1 };
    dim3 dimBlock { threadsPerBlock, 1, 1 };
    
    std::cout << "launching CUDA kernel with " << blocksPerGrid
              << " blocks of " << threadsPerBlock << " threads\n";

    AddVector<<<dimGrid, dimBlock>>>(vecSize, deviceVecA, deviceVecB, deviceVecC);
    cudaErr = ::cudaGetLastError();

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to launch AddVector kernel: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* 計算結果をデバイスからホストに転送 */
    cudaErr = ::cudaMemcpy(hostVecC, deviceVecC, vecSize * sizeof(float),
                           cudaMemcpyDeviceToHost);

    if (cudaErr != cudaError::cudaSuccess) {
        std::cerr << "failed to copy vector C from device to host: "
                  << ::cudaGetErrorString(cudaErr) << '\n';
        goto Cleanup;
    }

    /* 計算結果の検証 */
    for (int i = 0; i < vecSize; ++i) {
        if (std::fabs(hostVecA[i] + hostVecB[i] - hostVecC[i]) > 1e-5) {
            std::cerr << "result verification failed at element "
                      << i << '\n';
            goto Cleanup;
        }
    }

    std::cout << "vector addition succeeded\n";

Cleanup:
    /* デバイスのベクトル用のメモリ領域を解放 */
    if (deviceVecA != nullptr) {
        cudaErr = ::cudaFree(deviceVecA);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device vector A: "
                      << ::cudaGetErrorString(cudaErr) << '\n';
    }

    if (deviceVecB != nullptr) {
        cudaErr = ::cudaFree(deviceVecB);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device vector B: "
                      << ::cudaGetErrorString(cudaErr) << '\n';
    }

    if (deviceVecC != nullptr) {
        cudaErr = ::cudaFree(deviceVecC);

        if (cudaErr != cudaError::cudaSuccess)
            std::cerr << "failed to free device vector C: "
                      << ::cudaGetErrorString(cudaErr) << '\n';
    }

    /* ベクトル用のメモリ領域を解放 */
    if (hostVecA != nullptr)
        delete[] hostVecA;

    if (hostVecB != nullptr)
        delete[] hostVecB;

    if (hostVecC != nullptr)
        delete[] hostVecC;

    return EXIT_SUCCESS;
}
