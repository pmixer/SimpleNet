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

void quadCostFunc(Vector *output, Vector *det, int label) {
  for (int i = 0; i < output->len; i++) {
    if (i == label) {
      det->data[i] = output->data[i] - 1;
    }else {
      det->data[i] = output->data[i];
    }
  }
}

void mnistTest() {
  // Prepare data
  double **trainingData, **testData;
  int minTestPicNum = 100;
  readInMiniData(&trainingData, minTestPicNum);

  // Init the network
  SimpleNet myNet;
  int layerNum = 2;
  int *layerSizes = (int *)malloc(sizeof(int)*layerNum);
  layerSizes[0] = 724, layerSizes[1] = 10;
  initNetWork(&myNet, layerNum, layerSizes);

  // Params for learning, values below are kind of hand-tuned with no math directions which need improving
  double stepFactor = 0.1;
  int maxIteration = 50; //Epoch num, as they always set it to 50 in Currennt

  // Training by backprpagation
  for (int i = 0; i < maxIteration; i++) {
    clear(&myNet);
    for (int j = 0; j < minTestPicNum; j++) {
      int di = j; //data index
      // Test forward pass of the network
      forward(&myNet, trainingData[di]+1);
      // Test backward
      backward(&myNet, trainingData[di][0], &quadCostFunc, stepFactor);
    }
    update(&myNet);
  }

  // Debug by writing net weights
  //  writeMat(&(myNet.fls[0].weightDet), "fc1det");

  // Check nan error
  // printNet(&myNet);

 // Test overfitting result
  int right = 0;
  for (int j = 0; j < minTestPicNum; j++) {
    forward(&myNet, trainingData[j]+1);
    int res = selectFromOutput(&myNet);
    right += (trainingData[j][0] == (double)res);
    printf("label: %lf, res: %d\n", trainingData[j][0], res);
  }
  printf("Accuracy: %lf\n", right/(double)minTestPicNum);

}
int main()
{
  mnistTest();
  return 0;
}
