#ifndef SIMPLENET
#define SIMPLENET

#include "simplela.h"

struct SimpleNet;

struct InputLayer {
	struct Vector input;
};

struct ConnectionLayer {
	int inputsize;
	int outputsize;
	struct Vector bias, biasDet;
	struct Mat weight, weightDet;
	struct Vector res, det;
};

// Use for activation and softmax etc.
struct TransformLayer {
	struct Vector res;
	struct Vector det;
};

struct SimpleNet
{
	int hiddenLayerNum;
	struct InputLayer inputLayer;
	struct ConnectionLayer *fls;// fully connection layers
	struct TransformLayer *tls;// transformation layers
	struct Vector *output;
};

// Init the network, assign memory space to vectors and matrices
void initNetWork(struct SimpleNet *net, int layerNum, int *layerSize);

// Forward and compute the result as res value in Softmax/last layer
void forward(struct SimpleNet *net, double *input);

// Back propogation, directly update weights and bias,
// rigid and risky but effortless for no parallel computing needed
// and if SGD required, just use accumulative det array and add det matrix

void clear(struct SimpleNet *net);
//void reset(struct SimpleNet *net);
void backward(struct SimpleNet *net, int, void(*costFuncDet)(struct Vector *, struct Vector *, int), double sf);
void update(struct SimpleNet *net);

// Activation functions
double sigmoid(double num);
void acFun(struct Vector *act, struct Vector *output);
void softmax(struct Vector *input, struct Vector *output);

// Computing derivatives
double sigmoidDet(double num);
void acFunBack(struct Vector *input, struct Vector *res, struct Vector *det);
void softmaxBack(struct Vector *input, struct Vector *res, struct Vector *det);

// Determine which label it belongs to
int selectFirstBiggest(struct SimpleNet *net);
int selectFromOutput(struct SimpleNet * net);

// Cost Function

// Send derivative of cost function to last layer

#endif
