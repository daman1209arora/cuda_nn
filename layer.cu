#include "layer.h"
#include <cmath>
#include <random>
#include <stdlib.h>
#include <iostream>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

float uniform_real(float l, float r) {
    float rand = random() / RAND_MAX;
    return rand * (r - l) + l;
}

Layer::Layer() {}

Layer::Layer(int R, int C) {

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // S

    W = (float*)malloc(R * C * sizeof(float));
    b = (float*)malloc(C * sizeof(float));

    float std = 1.0f / sqrt(C);
    std::uniform_real_distribution<> dist(-std, std);

    for(int i = 0; i < R; i++) {
        for(int j = 0; j < C; j++)
            W[IDX2C(i, j, R)] = dist(gen);
    }

    for(int i = 0; i < C; i++) {
        b[IDX2C(i, 0, R)] = dist(gen);
    }
}

float* Layer::forward() {
    return NULL;
}