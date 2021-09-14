#include "nn.h"
#include "layer.h"

NN::NN() {}

NN::NN(const int hiddenLayerSizes[], int Fx, int Fy) {

    
    int numHiddenLayers = sizeof(hiddenLayerSizes) / sizeof(*hiddenLayerSizes);
    int numLayers = numHiddenLayers + 1;

    this->inputDim = inputDim;
    this->outputDim = outputDim;
    this->layers = (Layer**)malloc(numLayers * sizeof(Layer*));

    for(int i = 0; i < numHiddenLayers + 1; i++) {
        int inputDim = (i > 0) ? hiddenLayerSizes[i - 1] : Fx;
        int outputDim = (i < numHiddenLayers) ? hiddenLayerSizes[i] : Fy;
        layers[i] = new Layer(inputDim, outputDim);
    }
}

void NN::toGPU() {
    for(int i = 0; i < numLayers; i++) {
        layers[i]->toGPU();
    }
}