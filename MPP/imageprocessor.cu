#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_processor.cuh"


// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
    unsigned char *ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}

// Finds the resultant along a vector on the GPU
__global__ void resultant(unsigned char *a, unsigned char *b, unsigned char *c)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float opposite_side = float(a[idx]);
    float adjacent_side = float(b[idx]);

    // Figure out the hypotenuse 
    c[idx] = (unsigned char) sqrtf((opposite_side + adjacent_side)*(opposite_side + adjacent_side ) - (2 * opposite_side * adjacent_side));
    if ( c[idx] > 15 && c [idx -1] != 148 )
      c[idx] = 148;
    else
      c[idx] = 0;

}

__global__ void boxfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh)
{
    // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Variables to store the sum
    int count = 0;
    float sum = 0.0;

    // Do the blur operation by summing the surround pixels
    for(int j = -(bh/2); j <= (bh/2); j++)
    {
        for(int i = -(bw/2); i <= (bw/2); i++)
        {
            // Verify that this offset is within the image boundaries
            if((x+i) < iw && (x+i) >= 0 && (y+j) < ih && (y+j) >= 0)
            {
                sum += (float) source[((y+j) * iw) + (x+i)];
                count++;
            }
        }
    }

    // Average the sum
    sum /= (float) count;
    dest[(y * iw) + x] = (unsigned char) sum;
}


__global__ void sobelfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest)
{
    // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Operate only if we are in the correct boundaries
    if(x > 0 && x < iw - 1 && y > 0 && y < ih - 1)
    {
        int gx = -source[iw*(y-1)+(x-1)] + source[iw*(y-1)+(x+1)] +
                 -2*source[iw*(y)+(x-1)] + 2*source[iw*(y)+(x+1)] +
                 -source[iw*(y+1)+(x-1)] + source[iw*(y+1)+(x+1)];
        int gy = -source[iw*(y-1)+(x-1)] - 2*source[iw*(y-1)+(x)] 
                 -source[iw*(y-1)+(x+1)] +
                  source[iw*(y+1)+(x-1)] + 2*source[iw*(y+1)+(x)] +
                  source[iw*(y+1)+(x+1)];
        dest[iw*y+x] = (int) sqrt((float)(gx)*(float)(gx) + (float)(gy)*(float)(gy));
    }
}

void boxfilter(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh)
{
    // allocate memory for the bitmap in GPU memory
    unsigned char *dev_source, *dev_dest; 
    cudaHostGetDevicePointer(&dev_source, source, 0);
    cudaHostGetDevicePointer(&dev_dest, dest, 0);

    // Run the boxfilter kernel
    dim3 blocks(iw/16, ih/16);
    dim3 threads(16, 16); 

    // Execute the kernel
    boxfilter_kernel<<<blocks,threads>>>(iw, ih, dev_source, dev_dest, bw, bh);
    cudaThreadSynchronize();
}

void sobelfilter(int iw, int ih, unsigned char *source, unsigned char *dest)
{
    // allocate memory for the bitmap in GPU memory
    unsigned char *dev_source, *dev_dest; 
    cudaHostGetDevicePointer(&dev_source, source, 0);
    cudaHostGetDevicePointer(&dev_dest, dest, 0);

    // Run the boxfilter kernel
    dim3 blocks(iw/16, ih/16);
    dim3 threads(16, 16); 

    // Execute the kernel
    sobelfilter_kernel<<<blocks,threads>>>(iw, ih, dev_source, dev_dest);
    cudaThreadSynchronize();
}



