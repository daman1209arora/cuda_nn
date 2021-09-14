#ifndef __nn__
#define nn
#include "layer.h"
class NN {
public:
    int inputDim, outputDim;
    int numLayers;
    Layer** layers;

    NN();
    NN(const int hiddenLayerSizes[], int Fx, int Fy);
    void toGPU();
};

#endif