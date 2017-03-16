/*
    Simple linear algebra operation for the network
*/

#ifndef SIMPLELA
#define SIMPLELA
#include <stdbool.h>
struct Mat
{
    double **data;
    int rowNum;
    int colNum;
};

// Did not distinguish column vector and row vector
struct Vector
{
    double *data;
    int len;
};

double * getVecSpace(int size);
double ** getMatSpace(int inputLayerSize, int outputLayerSize);
void clearVector(struct Vector *vec);
void clearMat(struct Mat *mat);
void vplusv(struct Vector *vec, struct Vector *delta, double factor);
void vcpv(struct Vector *des, struct Vector *src);
void mplusm(struct Mat *m, struct Mat *dm, double factor);
void vmv(struct Vector *in_vec, struct Mat *mat, struct Vector *out_vec, bool mtrans);
void vvm(struct Vector *lvec, struct Vector *rvec, struct Mat *mat);
void printVector(struct Vector *vec);
#endif
