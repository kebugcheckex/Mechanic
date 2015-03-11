/*
    Video Content Analysis

    This file is part of PWICE project.

    Contributors:
    - Wayne Huang
    - Xu Qiu
    - Brian Lan
    - Xinyu Chen
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool DetectFace(Mat& inputImage, Mat& outputImage, CascadeClassifier& classifier)
{
    Mat frame, img_s, frame_gray;
	vector<Rect> objects;
	bool detected = false;
	while(inputVideo.read(frame))
	{
		total_frames++;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		if (!detected)
		{
            face_cascade.detectMultiScale(frame_gray, objects, 1.3, 5);
            cout << "Objects detected:" << objects.size() << "\t";
            if (objects.size() > 0) detected = true;
		}

		if (detected && frame_count < 10)
		{
            frame_count++;
            for(int i = 0; i < objects.size(); ++i)
                rectangle(frame, objects[i], Scalar(0,255,0), 2);
            cout << "Frame count" << frame_count++ << endl;
		}
		else
		{
            frame_count = 0;
            detected = false;
		}
    return false;
}

bool DetectText(Mat& inputImage, Mat& outputImage)
{
    return false;
}

int main(int argc, char** argv)
{
    if (argc < 3)   // We need two parameters, one for input file, and the other one for output file.
    {
        cout << "Usage: " << argv[0] << " input-file output-file" << endl;
        cout << "Input can be either a file on the disk or a URL to a video stream." << endl;
        return 1;
    }

    string inputFileName(argv[1]);
	VideoCapture vc(inputFileName);
	if (!vc.isOpened())
	{
        cerr << "Error: Failed to open the input file " << inputFileName << endl;
        return 2;
	}

	Mat inputFrame;
	vc >> inputFrame;
	int width = inputFrame.cols;
	int height = inputFrame.rows;

	string outputFileName(argv[2]);
	VideoWriter vw(argv[2], CV_FOURCC('M','P','E','G'), 25, Size(width*2, height*2));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}

	// TODO check whether the xml file exists
	CascadeClassifier face_cascade = CascadeClassifier("/home/xinyu/Videos/haarcascade_frontalface_default.xml");

	Mat outputFrame;
	int total_frames = 0;

	do {
        Mat faceResult;
        DetectFace(inputFrame, faceResult, face_cascade);
        Mat textResult;
        DetectText(inputFrame, textResult);
        /*
            The output image consists of four parts
            +-------+-------+
            |       |       |
            |   1   |   2   |
            |       |       |
            +-------+-------+
            |       |       |
            |   3   |   4   |
            |       |       |
            +-------+-------+
            1 - Original Image
            2 - Face Detection Result
            3 - Text Detection Result
            4 - Text Binary Image
        */
	} while (vc.read(inputFrame))
	cout << "Reach the end of the input file" << endl;
    cout << "Total frames = " << total_frame << endl;
    cout << "Done!" << endl;
	return 0;
}
