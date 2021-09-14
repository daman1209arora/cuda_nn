#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "cublas_v2.h"
#include "layer.h"
#include "nn.h"


#define IDX2C(i,j,ld) (((j)*(ld))+(i))

struct Dataset {
    float* X;
    float* Y;
    int N, Fx, Fy;
};

Dataset* load_dataset(std::string filename, int lim=-1) {
    std::ifstream ifs(filename.c_str());

    std::string dataset_info;
    getline(ifs, dataset_info);
    std::stringstream ss(dataset_info);

    std::string tok;
    ss >> tok;
    int N;
    if(lim == -1)
        N = atoi(tok.c_str());
    else
        N = min(atoi(tok.c_str()), lim);
    
    ss >> tok;
    int Fx = atoi(tok.c_str());
    ss >> tok;
    int Fy = atoi(tok.c_str());

    float* X = (float*)malloc(N * Fx * sizeof(float));
    float* Y = (float*)malloc(N * Fy * sizeof(float));

    for(int i = 0; i < N; i++) {
        std::string s;
        getline(ifs, s);
        std::stringstream sX(s);
        for(int j = 0; j < Fx; j++) {
            std::string tok;
            sX >> tok;
            X[IDX2C(i, j, N)] = atof(tok.c_str());
        }
        
        getline(ifs, s);
        std::stringstream sY(s);
        for(int j = 0; j < Fy; j++) {
            std::string tok;
            sY >> tok;
            Y[IDX2C(i, j, N)] = atof(tok.c_str());
        }
    }

    Dataset* ds = (Dataset*)malloc(sizeof(Dataset));
    ds->X = X; ds->Y = Y; ds->N = N; ds->Fx = Fx; ds->Fy = Fy;
    
    return ds;
}

__global__ void addBias(float* WXdevPtr, float* bias, int N, int M) {
    // Adds bias to WXdevPtr
    int idx = blockIdx.x * gridDim.x + threadIdx.x;
    int biasIdx = (idx / (N / M));
    if((idx < N) && biasIdx < M) {
        WXdevPtr[idx] += bias[biasIdx];
    }
}

__global__ void relu(float* a, float* b, int N) {
    int idx = blockIdx.x * gridDim.x + threadIdx.x;
    if(idx < N) {
        if(a[idx] < 0)
            b[idx] = 0;
        else
            b[idx] = a[idx];
    }
}

int main(int argc, char* argv[]) {
    Dataset* ds = load_dataset("datasets/two-spiral.train", 5);
    
    int N = ds->N, Fx = ds->Fx, Fy = ds->Fy;
    float* X = ds->X;

    const int layerSizes[] = {3, 2};
    NN model = NN(layerSizes, Fx, Fy);
    
    float* XdevPtr = nullptr;
    
    cudaMalloc((void**)&XdevPtr, N * Fx * sizeof(float));
    cublasHandle_t handleX;
    cublasCreate(&handleX);
    cublasSetMatrix(N, Fx, sizeof(float), X, N, XdevPtr, N);
    
    
    /*cudaFree(XdevPtr);
    cublasDestroy(handleX);
    free(ds->X); 
    free(ds->Y);
    */
}