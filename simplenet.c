#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <unistd.h>
#include "simplenet.h"

void initNetWork(struct SimpleNet *net, int layerNum, int *layerSize)
{
  int inputSize = layerSize[0];
  net->inputLayer.input.len = inputSize;
  net->inputLayer.input.data = getVecSpace(inputSize);
  //net->inputLayer.input->data = getVecSpace(inputSize);

  int hiddenLayerNum = layerNum - 1;// Number of hidden layers
  net->hiddenLayerNum = hiddenLayerNum;
  net->fls = (struct ConnectionLayer *)malloc(sizeof(struct ConnectionLayer)*hiddenLayerNum);
  net->tls = (struct TransformLayer *)malloc(sizeof(struct TransformLayer)*hiddenLayerNum);

  // Get space for vectors and matrices
  for (int i = 0; i < hiddenLayerNum; i++) {
    net->fls[i].inputsize = layerSize[i];
    net->fls[i].outputsize = layerSize[i+1];

    int outputsize = layerSize[i+1];
    net->fls[i].bias.len = outputsize;
    net->fls[i].bias.data = getVecSpace(outputsize);
    net->fls[i].biasDet.len = outputsize;
    net->fls[i].biasDet.data = getVecSpace(outputsize);
    net->fls[i].weight.rowNum = layerSize[i], net->fls[i].weight.colNum = outputsize;
    net->fls[i].weight.data = getMatSpace(layerSize[i], outputsize);
    net->fls[i].weightDet.rowNum = layerSize[i], net->fls[i].weightDet.colNum = outputsize;
    net->fls[i].weightDet.data = getMatSpace(layerSize[i], outputsize);

    net->fls[i].res.len = net->fls[i].det.len = outputsize;
    net->fls[i].res.data = getVecSpace(outputsize);
    net->fls[i].det.data = getVecSpace(outputsize);

    net->tls[i].res.len = net->tls[i].det.len = outputsize;
    net->tls[i].res.data = getVecSpace(outputsize);
    net->tls[i].det.data = getVecSpace(outputsize);
  }
  net->output = &(net->tls[hiddenLayerNum-1].res);
}

//FowardPass with sigmoid included
void forward(struct SimpleNet *net, double *input) {
  net->inputLayer.input.data = input;
  //dirty fixation
  // for (int i = 0; i < net->inputLayer.input.len; i++) {
  //   net->inputLayer.input.data[i] = input[i];
  // }
  //printVector(&(net->inputLayer.input));

  vmv(&(net->inputLayer.input), &(net->fls[0].weight), &(net->fls[0].res), false);
  vplusv(&(net->fls[0].res), &(net->fls[0].bias), 1.0);

  // Before softmax
  for (int i = 0; i + 1 < net->hiddenLayerNum; i++) {
    acFun(&(net->fls[i].res), &(net->tls[i].res));
    vmv(&(net->tls[i].res), &(net->fls[i+1].weight), &(net->fls[i+1].res), false);
    vplusv(&(net->fls[i+1].res), &(net->fls[i+1].bias), 1.0);
  }

  // softmax
  int lli = net->hiddenLayerNum-1;// last layer index
  softmax(&net->fls[lli].res, &net->tls[lli].res);
}

void clear(struct SimpleNet *net) {
  for (int i = 0; i < net->hiddenLayerNum; i++) {
    clearVector(&(net->fls[i].biasDet));
    clearMat(&(net->fls[i].weightDet));
    clearVector(&(net->tls[i].det));
  }
}

// det is derivative vector passed from cost function
void backward(struct SimpleNet *net, int label, void(*costFunDet)(struct Vector *output, struct Vector *det, int label)) {
  int li = net->hiddenLayerNum-1;//last layer index
  // set derivative function to softmax layer det vector
  costFunDet(net->output, &(net->tls[li].det), label);

  for (; li > 0; li--) {
    // consequently, softmaxBack is same as acFunBack using sigmoid
    softmaxBack(&(net->tls[li].det), &(net->tls[li].res), &(net->fls[li].det));
    // To update weight det and bias det
    vvm(&(net->tls[li-1].res), &(net->fls[li].det), &(net->fls[li].weightDet));// vector multiple vector to matrix, for weight matrix det
    //vcpv(&(net->fls[li].biasDet), &(net->fls[li].det));// vector copy another vector's value, for bias det
    vplusv(&(net->fls[li].biasDet), &(net->fls[li].det), 1.0);
    vmv(&(net->fls[li].det), &(net->fls[li].weight) ,&(net->tls[li-1].det), true);
  }
  // last hidden layer, now li  = 0
  softmaxBack(&(net->tls[li].det), &(net->tls[li].res), &(net->fls[li].det));
  vvm(&(net->inputLayer.input), &(net->fls[li].det), &(net->fls[li].weightDet));// vector multiple vector to matrix, for weight matrix det
  vcpv(&(net->fls[li].biasDet), &(net->fls[li].det));// vector copy another vector's value, for bias det
}

// update using a step factor
void update(struct SimpleNet *net, double sf) {
  for (int i = 0; i < net->hiddenLayerNum; i++) {
    mplusm(&(net->fls[i].weight), &(net->fls[i].weightDet), -sf);
    vplusv(&(net->fls[i].bias), &(net->fls[i].biasDet), -sf);
  }
}

void softmax(struct Vector *input, struct Vector *output) {
  double sum = 0;
  for (int i = 0; i < input->len; i++) {
    double tmp = exp(input->data[i]);
    output->data[i] = tmp;
    sum += tmp;
  }

  for (int i = 0; i < input->len; i++) {
    output->data[i] /= sum;
  }
}

void softmaxBack(struct Vector *inputdet, struct Vector *inputres, struct Vector *det) {
  for (int i = 0; i < inputdet->len; i++) {
    det->data[i] += inputdet->data[i]*sigmoidDet(inputres->data[i]);
  }
}

void acFun(struct Vector *act, struct Vector *output)
{
    // Activation part could be splitted into a more powerful module to deal with many kinds of forward pass and bp in the future
    for (int ctr = 0; ctr < act->len; ctr++)
    {
        output->data[ctr] = sigmoid(act->data[ctr]);
    }
}

void acFunBack(struct Vector *inputdet, struct Vector *inputres, struct Vector *det) {
  softmaxBack(inputdet, inputres, det);
}

double sigmoid(double num)
{
    return 1.0/(1+exp(-num));
}

double sigmoidDet(double num)
{
    return num*(1-num);
}

int selectFromOutput(struct SimpleNet * net)
{
    return selectFirstBiggest(net);
}

int selectFirstBiggest(struct SimpleNet *net)
{
    int counter2; // counter2 indicates the code is pasted from another episode
    int maxAt = 0;
    double maxAmongOutput = net->tls[1].res.data[0];
    printf("\nSelecting from: ");
    for (counter2 = 0; counter2 < net->tls[1].res.len; counter2++)
    {
        printf("%lf ", net->tls[1].res.data[counter2]);
        if (net->tls[1].res.data[counter2] > maxAmongOutput)
        {
            maxAmongOutput = net->tls[1].res.data[counter2];
            maxAt = counter2;//Or just use maxAt, delete maxAmongOutput
        }
    }
    printf("\n");
    return maxAt;
}
