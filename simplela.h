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
void mplusm(struct Mat *m, struct Mat *dm, double factor);
void vmv(struct Vector *in_vec, struct Mat *mat, struct Vector *out_vec, bool mtrans);
#endif
