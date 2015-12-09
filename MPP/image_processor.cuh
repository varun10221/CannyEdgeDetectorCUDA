#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef long ssize_t;

// GPU constant memory to hold our kernels (extremely fast access time)
__constant__ float convolutionKernel[256];


void boxfilter(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh);
void sobelfilter(int iw, int ih, unsigned char *source, unsigned char *dest);
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr);
__global__ void resultant(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void sobelfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest);
__global__ void boxfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh);

//__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination);