#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float dataset[][2] = {{0, 0},
                      {1, 2},
                      {2, 4},
                      {3, 6},
                      {4, 8}};
#define numDataset (sizeof(dataset)/sizeof(dataset[0]))


float rand_float(void){return (float)rand() / (float)RAND_MAX;}


float loss(float w, float b){
    float mse = 0.0f;
    for (size_t i = 0; i < numDataset; i++){
        float x = dataset[i][0];
        float y = x*w + b;
        float d = y - dataset[i][1];
        mse += d*d;  // Measure the distance between ideal and actual 
        // printf("actual: %f, expected: %f \n", y, dataset[i][1]);
    }
    mse /= numDataset;
    return mse;
}


void main(){
    srand(69);

    float w = rand_float()*100.0f; // Model: y = x*w;
    float b = rand_float()*5.0f;
    
    float lr = 1e-3;
    float eps = 1e-3;

    printf("%f\n", loss(w, b));
    for (size_t i = 0; i < 10000; i++){
        float dw = (loss(w + eps, b) - loss(w, b))/eps;  // Finite difference for derivative approximation
        float db = (loss(w, b + eps) - loss(w, b))/eps;
        w -= lr*dw;
        b -= lr*db;
        printf("Loss = %f, w = %f, b = %f\n", loss(w, b), w, b);
    }
    printf("---------------------------------\n");
    printf("w = %f, b = %f\n", w, b);
}
