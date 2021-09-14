#ifndef __layer__
#define __layer__

class Layer {
public:
    Layer();
    Layer(int r, int c);
    float* W; // r * c matrix
    float* b; // c * 1 matrix
    
    float* WdevPtr = nullptr;
    float* bdevPtr = nullptr;

    void toGPU();
    float* forward();
};

#endif