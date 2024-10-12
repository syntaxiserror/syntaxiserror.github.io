#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OR-gate
float dataset[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};
#define numDataset (sizeof(dataset)/sizeof(dataset[0]))


float sigmoidf(float x){return 1.f/(1.f + expf(-x));}


float rand_float(void){return (float)rand() / (float)RAND_MAX;}


// Do the forward pass and calculate the MSE cost function
float loss(float w1, float w2, float b){
    float mse = 0.0f;
    for (size_t i = 0; i < numDataset; i++){
        float x1 = dataset[i][0];
        float x2 = dataset[i][1];
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        float d = y - dataset[i][2];
        mse += d*d; 
    }
    mse /= numDataset;
    return mse;
}


void main(){
    srand(69);
    float w1 = rand_float()*100.0f;
    float w2 = rand_float();
    float b = rand_float()*10.0f;
    int epochs = 3000*100;
    float eps = 1e-3;
    float lr = 1e-1;

    printf("w1 = %f, w2 = %f, b = %f", w1, w2, b);

    // Calculate derivative for error
    // Subtract the gradient for each of two weights
    printf("%f\n", loss(w1, w2, b));
    for (size_t i = 0; i < epochs; i++){
        float dw1 = (loss(w1 + eps, w2, b) - loss(w1, w2, b))/eps;
        float dw2 = (loss(w1, w2 + eps, b) - loss(w1, w2, b))/eps;
        float db = (loss(w1, w2, b + eps) - loss(w1, w2, b))/eps;
        w1 -= lr*dw1;
        w2 -= lr*dw2;
        b -= lr*db;
        printf("w1 = %f, w2 = %f, b = %f, loss = %f\n", w1, w2, b, loss(w1, w2, b));
    }
}
