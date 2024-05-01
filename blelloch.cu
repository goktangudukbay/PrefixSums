#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>


#define NO_OF_THREADS_IN_BLOCK 1024
#define NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES NO_OF_THREADS_IN_BLOCK*2
#define NO_OF_BLOCKS 2048
#define N NO_OF_BLOCKS*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES

__global__ void prefixSum(float *g_odata, float *g_idata, float* sum, int n) { 
    __shared__ float temp[NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES];  // allocated on invocation 

    int thid = threadIdx.x; 
    int offset = 1;

    temp[2*thid] = g_idata[blockIdx.x*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES+2*thid]; // load input into shared memory 
    temp[2*thid+1] = g_idata[blockIdx.x*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES+2*thid+1];

    for (int d = n>>1; d > 0; d >>= 1)                    
    // build sum in place up the tree 
    { 
        __syncthreads();    
        
        if (thid < d){ 

            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;  

            temp[bi] += temp[ai];        
        }    
        
        offset *= 2; 
    }


    if (thid == 0) { // clear the last element
        sum[blockIdx.x] = temp[n-1];
        temp[n - 1] = 0; 
    } 


    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
    {      
        offset >>= 1;      
        __syncthreads();     
    
        if (thid < d){ 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1; 
        
            float t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    } 

    __syncthreads();

    g_odata[blockIdx.x*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES+2*thid] = temp[2*thid]; // write results to device memory      
    g_odata[blockIdx.x*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES+2*thid+1] = temp[2*thid+1];
}

//add sum[block_index] to g_odata
__global__ void addKernel(float *g_odata, float *sum){
    if(blockIdx.x > 0){
        g_odata[blockIdx.x*NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES+threadIdx.x] += sum[blockIdx.x];
    }
}

int main(){
    float *g_odata, *g_idata, *sum, *cumSum,*var;

    cudaMallocManaged(&g_odata, N*sizeof(float));
    cudaMallocManaged(&g_idata, N*sizeof(float));
    cudaMallocManaged(&sum, NO_OF_BLOCKS*sizeof(float));
    cudaMallocManaged(&cumSum, NO_OF_BLOCKS*sizeof(float));
    cudaMallocManaged(&var, sizeof(float));
    
    cumSum[0] = 0;

    for(int i = 0; i < N; i++)
        g_idata[i] = i;

    cudaMemPrefetchAsync(g_idata, N*sizeof(float), 0);      
    cudaMemPrefetchAsync(g_odata, N*sizeof(float), 0);       
    cudaMemPrefetchAsync(sum, NO_OF_BLOCKS*sizeof(float), 0);       
    cudaMemPrefetchAsync(cumSum, NO_OF_BLOCKS*sizeof(float), 0);        
    cudaMemPrefetchAsync(var, sizeof(float), 0);

    auto start = std::chrono::system_clock::now();

    prefixSum<<<NO_OF_BLOCKS, NO_OF_THREADS_IN_BLOCK>>>(g_odata, g_idata, sum, NO_OF_ELEMENTS_EACH_BLOCK_PROCESSES);

    cudaDeviceSynchronize();

    if(NO_OF_BLOCKS > 1){
        prefixSum<<<1, NO_OF_THREADS_IN_BLOCK>>>(cumSum, sum, var, NO_OF_BLOCKS);
        cudaDeviceSynchronize();

        addKernel<<<NO_OF_BLOCKS, NO_OF_THREADS_IN_BLOCK*2>>>(g_odata, cumSum);
        cudaDeviceSynchronize();        
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Total Number of Elements (N): " << N << "\n Number of Blocks: " << NO_OF_BLOCKS << "\n Number of Threads in Each Block: " << NO_OF_THREADS_IN_BLOCK << "\n Time: " << elapsed.count() << std::endl;

    cudaFree(g_odata);
    cudaFree(g_idata);
    cudaFree(sum);
    cudaFree(cumSum);
    cudaFree(var);

    return 0;
}
