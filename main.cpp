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
	vector<Rect> objects;
	bool detected = false;
	outputImage = inputImage;
	Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    classifier.detectMultiScale(grayImage, objects, 1.3, 5);
    cout << "Objects detected:" << objects.size() << "\t";
    for(int i = 0; i < objects.size(); ++i)
        rectangle(outputImage, objects[i], Scalar(0,255,0), 2);
    return true;
}

bool DetectText(Mat& inputImage, Mat& outputImage, Mat& binaryImage)
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
    VideoCapture vc;
    if (inputFileName == "0")
        vc = VideoCapture(0);
    else
        vc = VideoCapture(inputFileName);

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

	Mat outputFrame(width*2, height*2, CV_8UC3);
	cout << "Debug: width=" << width << "\theight=" << height << endl;
    Mat faceResult(width, height, CV_8UC3);
    Mat textResult(width, height, CV_8UC3);
    Mat textBinary(width, height, CV_8UC3);
	int total_frames = 0;
	namedWindow("Result");
	do {

        DetectFace(inputFrame, faceResult, face_cascade);
        DetectText(inputFrame, textResult, textBinary);
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
        inputFrame.copyTo(outputFrame(Rect(0, 0, width, height)));
        faceResult.copyTo(outputFrame(Rect(width, 0, width, height)));
        textResult.copyTo(outputFrame(Rect(0, height, width, height)));
        textBinary.copyTo(outputFrame(Rect(width, height, width, height)));
        imshow("Result", outputFrame);
        if (waitKey(20) >= 0) break;
	} while (vc.read(inputFrame));
	cout << "Reach the end of the input file" << endl;
    cout << "Total frames = " << total_frames << endl;
    cout << "Done!" << endl;
	return 0;
}
