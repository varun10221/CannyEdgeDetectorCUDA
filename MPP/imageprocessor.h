#include <iostream>
#include <string>
#include <stdio.h>
#include <cstddef>
#include <sys/types.h>
#include <cuda.h>
#include<cuda_runtime.h>

typedef long ssize_t;

// GPU constant memory to hold our kernels (extremely fast access time)


/* A convolution function */
__global__ void convolve(unsigned char *, int, int, int, int, ssize_t, int, int, unsigned char *, float arr[]); 
__global__ void resultant(unsigned char *, unsigned char *, unsigned char *);
unsigned char* createImageBuffer(unsigned int, unsigned char **);
