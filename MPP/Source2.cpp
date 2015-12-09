#include <cstdio>
#include <iostream>
#include <conio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\opencv.hpp>
#include <windows.h>

using  namespace std;

using namespace cv;


int main(int, char**)
{

    VideoCapture cameracapture(0); // open the default camera
    if(!cameracapture.isOpened())  // check if we succeeded
        return -1;
	    cameracapture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cameracapture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    Mat Outputwindow;
    namedWindow("OUTPUT",1);

		int x; 

		for (;;)
		{
        Mat frame;
        cameracapture >> frame; // get a new frame from camera
		cvtColor(frame, Outputwindow, CV_BGR2GRAY);
		cout << "You have pressed the up key" << endl;

		GaussianBlur(Outputwindow, Outputwindow, Size(7,7), 1.5, 1.5);
        Canny(Outputwindow, Outputwindow, 0, 25, 3);
        imshow("OUTPUT", Outputwindow);
        if(waitKey(30) >= 0) break;
		}
		
		
    return 0;
}