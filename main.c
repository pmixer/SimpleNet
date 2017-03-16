/*This program is a demostration of MNIST recognition task implemented in C programming language*/
//No bias included in this version of the program
//This is a fast demo of the system, more work need to be done decomposing the parts of the net to make it accepting constructing the net
//with different kind of layers, deveritive matrix is essential to be included into it

// It's better to take hidden layers as an array of matrixs begin modifying codes - 2015-9-5
//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <unistd.h>
#include "simplenet.h"

void quadCostFunc(struct Vector *output, struct Vector *det, int label) {
  for (int i = 0; i < output->len; i++) {
    if (i == label) {
      det->data[i] = output->data[i] - 1;
    }else {
      det->data[i] = output->data[i];
    }
  }
}

int main()
{
    // Prepare data
    double **trainingData, **testData;

    int minTestPicNum = 100;
    readInMiniData(&trainingData, minTestPicNum);

    // Init the network
    struct SimpleNet myNet;
    // int inputSize = 724;
    int layerNum = 3;
    int *layerSizes = (int *)malloc(sizeof(int)*layerNum);
    layerSizes[0] = 724, layerSizes[1] = 100, layerSizes[2] = 10;
    initNetWork(&myNet, layerNum, layerSizes);

    // Params for learning, values below are kind of hand-tuned with no math directions which need improving
    double stepFactor = 0.00001, minorDiff = 0.0001; //Set the parameter for M(i,j) = M(i,j) - stepParam*(Partial Derivative)

    int maxIteration = 10000; //As they always set it to 50 in Currennt

    // Training by backprpagation
    for (int i = 0; i < maxIteration; i++) {
      int di = i%minTestPicNum; //data index
      // Test forward pass of the network
      forward(&myNet, trainingData[di]+1);
      int res = selectFromOutput(&myNet);
      printf("label: %lf, res: %d\n", trainingData[di][0], res);
      // Test backward
      clear(&myNet);
      backward(&myNet, trainingData[di][0], &quadCostFunc);
      update(&myNet, stepFactor);
    }
    //printf("\n%lf\n",(double)bp(&myNet, trainingData, minTestPicNum, stepFactor));
    return 0;
}
