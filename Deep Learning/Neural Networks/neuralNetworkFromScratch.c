#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple NN for XOR value predictions

double sigmoid(double x) {return 1 / (1 + exp(-x));}
double dSigmoid(double x) {return x * (1 - x);}  // Needed for backpropagation

double init_weights() {return ((double)rand()) / ((double)RAND_MAX);}

// Fisher-Yates Shuffle
void shuffle(int *array, size_t n){
    if (n > 1){
        size_t i;
        for (i = 0; i < n - 1; i++){
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHidden 2
#define numOutputs 1
#define numTrainingSets 4

int main() {
    const double lr = 0.1f;

    // Layers
    double hiddenLayer[numHidden];
    double outputLayer[numOutputs];

    // Biases
    double hiddenLayerBias[numHidden];
    double outputLayerBias[numOutputs];

    // Weights
    double hiddenLayerWeigths[numInputs][numHidden];
    double outputLayerWeigths[numHidden][numOutputs];

    // Dataset
    double features[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                   {0.0f, 1.0f},   
                                                   {1.0f, 0.0f},
                                                   {1.0f, 1.0f}};

    double labels[numTrainingSets][numOutputs] = {{0.0f},
                                                  {1.0f},   
                                                  {1.0f},
                                                  {0.0f}};

    // Initialize weights and biases
    for (int i=0; i < numInputs; i++){
        for (int j=0; j < numHidden; j++){
            hiddenLayerWeigths[i][j] = init_weights();
        }
    }

    for (int i=0; i < numHidden; i++){
        for (int j=0; j < numOutputs; j++){
            outputLayerWeigths[i][j] = init_weights();
        }
    }

    for (int i=0; i < numOutputs; i++){
        outputLayerBias[i] = init_weights();
    }

    // Training loop
    int trainingSetOrder[] = {0, 1, 2, 3};
    int numEpochs = 10000;

    // Forwardpass
    for (int epoch = 0; epoch < numEpochs; epoch++){
        shuffle(trainingSetOrder, numTrainingSets);
        for (int x = 0; x < numTrainingSets; x++){
            int i = trainingSetOrder[x];
            // Hidden layer
            for (int j = 0; j < numHidden; j++){
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++){
                    activation += features[i][k] * hiddenLayerWeigths[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            // Output layer
            for (int j = 0; j < numOutputs; j++){
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHidden; k++){
                    activation += hiddenLayer[k] * outputLayerWeigths[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            printf("Input: %g  Output: %g  Predicted Output: %g \n", 
                   features[i][0], features[i][1], outputLayer[0], labels[i][0]);

            // Backpropagation
            // Output layer
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++){
                double error = (labels[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }
            // Hidden layer
            double deltaHidden[numHidden];
            for (int j = 0; j < numHidden; j++){
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputLayerWeigths[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Apply change in output weights
            for (int j = 0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHidden; k++){
                    outputLayerWeigths[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            // Apply change in hidden weights
            for (int j = 0; j < numHidden; j++){
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++){
                    hiddenLayerWeigths[k][j] += features[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
}   