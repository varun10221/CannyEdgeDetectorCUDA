
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_processor.cuh"

/**
 * Convolution function for cuda.  Destination is expected to have the same width/height as source, but there will be a border
 * of floor(kWidth/2) pixels left and right and floor(kHeight/2) pixels top and bottom
 *
 * @param source      Source image host pinned memory pointer
 * @param width       Source image width
 * @param height      Source image height
 * @param paddingX    source image padding along x 
 * @param paddingY    source image padding along y
 * @param kOffset     offset into kernel store constant memory 
 * @param kWidth      kernel width
 * @param kHeight     kernel height
 * @param destination Destination image host pinned memory pointer
 */
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
    // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float sum = 0.0f;
    int   pWidth = kWidth/2;
    int   pHeight = kHeight/2;

    // Execute for valid pixels
    if(x >= pWidth+paddingX &&
       y >= pHeight+paddingY &&
       x < (gridDim.x * blockDim.x)-pWidth-paddingX &&
       y < (gridDim.y *blockDim.y )-pHeight-paddingY)
    {
        for(int j = -pHeight; j <= pHeight; j++)
        {
            for(int i = -pWidth; i <= pWidth; i++)
            {
                // Sample the weight for this location
                int ki = (i+pWidth);
                int kj = (j+pHeight);
                float w  = convolutionKernel[(kj * kWidth) + ki + kOffset];


                sum += w * float(source[((y+j) * width) + (x+i)]);
            }
        }
    }

    // Average the sum
    destination[(y * width) + x] = (unsigned char) sum;
}



int main (int argc, char** argv)
{
    // Open a webcamera
    cv::VideoCapture camera(0);
    cv::Mat          frame;
    if(!camera.isOpened()) 
        return -1;

    // Create the capture windows
    cv::namedWindow("Source");
    cv::namedWindow("Grayscale");
    cv::namedWindow("Blurred");
    cv::namedWindow("Gaussian");
    cv::namedWindow ("Box");
    cv::namedWindow ("sobel");
   
     // Create the gaussian kernel 
     // Credits, gaussian generation was created using the code from :https://sgsawant.wordpress.com/2009/11/05/generation-of-gaussian-kernel-mask/.

    const float gaussianKernel5x5[25] = 
    {
        4.f/240.f,   8.f/240.f,  10.f/240.f,   8.f/240.f, 4.f/240.f,   
        8.f/240.f,   18.f/240.f, 24.f/240.f,   18.f/240.f, 8.f/240.f,   
        10.f/240.f,  24.f/240.f, 30.f/240.f,   24.f/240.f, 10.f/240.f,   
        8.f/240.f,   18.f/240.f, 24.f/240.f,   18.f/240.f, 8.f/240.f,   
        4.f/240.f,   16.f/240.f, 10.f/240.f,    8.f/240.f,  4.f/240.f,   
    };
    cudaMemcpyToSymbol(convolutionKernel, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);


    // Sobel gradient kernels
    /* Sobel Gradient kernel source :http://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm */
   // Note: Angle is taken as zero.
    const float sobelGX[9] =
    {
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f,
    };
    const float sobelGY[9] =
    {
        1.f, 2.f, 1.f,
        0.f, 0.f, 0.f,
        -1.f, -2.f, -1.f,
    };
    cudaMemcpyToSymbol(convolutionKernel, sobelGX, sizeof(sobelGX), sizeof(gaussianKernel5x5));
    cudaMemcpyToSymbol(convolutionKernel, sobelGY, sizeof(sobelGY), sizeof(gaussianKernel5x5) + sizeof(sobelGX));
    const ssize_t sobelGradientXOffset = sizeof(gaussianKernel5x5)/sizeof(float);
    const ssize_t sobelGradientYOffset = sizeof(sobelGX)/sizeof(float) + sobelGradientXOffset;

    // Create CPU/GPU shared images - one for the initial and one for the result
    camera >> frame;
    unsigned char *sourceDataDevice, *blurredDataDevice, *edgesDataDevice,*boxDataDevice,*sobelDataDevice;
    cv::Mat source  (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
    cv::Mat blurred (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
    cv::Mat edges   (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));
    cv::Mat Imagebuffer2 (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &boxDataDevice));
    cv::Mat Imagebuffer3 (frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sobelDataDevice));

    // Create two temporary images (for holding sobel gradients)
    unsigned char *deviceGradientX, *deviceGradientY;
    cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
    cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);

    // Loop while capturing images
    while(1)
    {
        // Capture the image from the camera object;
        camera >> frame;
        // convert the image to gray scale
        cv::cvtColor(frame, source, CV_BGR2GRAY);

        // Record the time it takes to process

        {
            // convolution kernel launch parameters
            dim3 cblocks (frame.size().width / 16, frame.size().height / 16);
            dim3 cthreads(16, 16);

            // pythagorean kernel (resultant vector) launch paramters
            dim3 pblocks (frame.size().width * frame.size().height / 256);
            dim3 pthreads(256, 1);

            boxfilter(frame.size().width, frame.size().height, source.data, Imagebuffer2.data, 3, 3);

	     	sobelfilter(frame.size().width, frame.size().height, Imagebuffer2.data, Imagebuffer3.data);

            // Perform the gaussian blur (first kernel in store @ 0)
            convolve<<<cblocks,cthreads>>>(sourceDataDevice, frame.size().width, frame.size().height, 0, 0, 0, 5, 5, blurredDataDevice);

            // Perform the sobel gradient convolutions (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
            convolve<<<cblocks,cthreads>>>(blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
            convolve<<<cblocks,cthreads>>>(blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
            resultant<<<pblocks,pthreads>>>(deviceGradientX, deviceGradientY, edgesDataDevice);

            cudaThreadSynchronize ();
        }



        // Show the results
        cv::imshow("Source", frame);
        cv::imshow("Greyscale", source);
        cv::imshow("Blurred", blurred);
        cv::imshow("Gaussian", edges);
        cv::imshow("BOX",Imagebuffer2);
        cv::imshow("sobel",Imagebuffer3);

        // Spin
        if(cv::waitKey(1) == 27) break;
    }

    // Exit
    cudaFreeHost(source.data);
    cudaFreeHost(blurred.data);
    cudaFreeHost(edges.data);
    cudaFree(deviceGradientX);
    cudaFree(deviceGradientY);

    return 0;
}