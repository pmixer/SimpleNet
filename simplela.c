#include "simplela.h"
#include <stdio.h>

double * getVecSpace(int size) {
	double *vec = (double *)malloc(sizeof(double)*size);
	for (int i = 0; i < size; i++) {
		vec[i] = 0;
	}
	return vec;
}

double ** getMatSpace(int inputLayerSize, int outputLayerSize)
{
    double **mat = (double **)malloc(sizeof(double *)*inputLayerSize);
    int rowIndex, colIndex;
    for (rowIndex = 0; rowIndex < inputLayerSize; rowIndex++)
    {
        mat[rowIndex] = (double *)malloc(sizeof(double)*outputLayerSize);
        for (colIndex = 0; colIndex < outputLayerSize; colIndex++)
        {
            //Initialization is very important process and I still haven't released the power of statistics
            // mat[rowIndex][colIndex] = ((random() - RAND_MAX/2)*0.01)/(6*RAND_MAX);
						mat[rowIndex][colIndex] = 0;
            // printf("Got %lf ", mat[rowIndex][colIndex]);
        }
    }
    return mat;
}

void clearVector(struct Vector *vec) {
	for (int i = 0; i < vec->len; i++) {
		vec->data[i] = 0;
	}
}

void clearMat(struct Mat *mat) {
	for (int i = 0; i < mat->rowNum; i++) {
		for (int j = 0; j < mat->colNum; j++) {
			mat->data[i][j] = 0;
		}
	}
}

void vplusv(struct Vector *vec, struct Vector *delta, double factor) {
	for (int i = 0; i < vec->len; i++) {
		vec->data[i] += factor*delta->data[i];
	}
}

void vcpv(struct Vector *des, struct Vector *src) {
	for (int i = 0; i < des->len; i++) {
		des->data[i] = src->data[i];
	}
}

void mplusm(struct Mat *m, struct Mat *dm, double factor) {
	for (int i = 0; i < m->rowNum; i++) {
		for (int j = 0; j < m->colNum; j++) {
			m->data[i][j] += factor*dm->data[i][j];
		}
	}
}

void vmv(struct Vector *in_vec, struct Mat *mat, struct Vector *out_vec, bool mtrans)
{
    // Row num is equal to v1 length and column num is equal to v2 length of the given matrix
	// if ((in_vec->len != mat->rowNum)||(out_vec->len != mat->colNum))
	// {
	// 	printf("Runtime error, invalid vector times matrix assigning to another vector\n");
	// 	printf("vec1 len: %d, matrix size: %d %d, vec2 len: %d\n", in_vec->len, mat->rowNum, mat->colNum, out_vec->len);
	// 	exit(1);
	//
	// }

		// how to be execution efficient and code sufficient?
		// look at code below
    int counter1, counter2;
		if (!mtrans) {
			for (counter1 = 0; counter1 < out_vec->len; counter1++)
			{
					double tmp = 0;
					for (counter2 = 0; counter2 < in_vec->len; counter2++)
					{
							tmp += in_vec->data[counter2]*mat->data[counter2][counter1];
					}
					out_vec->data[counter1] = tmp;
			}
		}else {
			for (counter1 = 0; counter1 < out_vec->len; counter1++)
	    {
	        double tmp = 0;
	        for (counter2 = 0; counter2 < in_vec->len; counter2++)
	        {
	            tmp += in_vec->data[counter2]*mat->data[counter1][counter2];
	        }
	        out_vec->data[counter1] = tmp;
	    }
		}
}

void vvm(struct Vector *lvec, struct Vector *rvec, struct Mat *mat) {
	for (int i = 0; i < lvec->len; i++) {
		for (int j = 0; j < rvec->len; j++) {
			mat->data[i][j] = lvec->data[i]*rvec->data[j];
		}
	}
}
