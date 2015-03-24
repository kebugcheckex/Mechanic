/*
    Video Content Analysis

    This file is part of PWICE project.

    Contributors:
    - Wayne Huang
    - Xu Qiu
    - Brian Lan
    - Xinyu Chen
*/

#include "stdafx.h"
#include "FaceDetector.h"
#include "VideoReader.h"

using namespace std;
using namespace cv;
using namespace tbb;


int main(int argc, char** argv)
{
    if (argc < 3)
    { // We need two parameters, one for input file, and the other one for output file.
        cout << "Usage: " << argv[0] << " input-file output-file" << endl;
        cout << "Input can be either a file on the disk or a URL to a video stream." << endl;
        return 1;
    }

	concurrent_queue<Mat> faceFrameBuffer;
	concurrent_queue<Mat> textFrameBuffer;
	concurrent_queue<Mat> binaryFrameBuffer;

    string inputFileName(argv[1]);
    FaceDetector faceDetector;
    VideoReader videoReader(inputFileName, &faceDetector);

    int width = videoReader.GetFrameSize().width;
    int height = videoReader.GetFrameSize().height;
    cout << "Width = " << width << " Height = " << height << endl;
	string outputFileName(argv[2]);
	// TODO - FOURCC should not be hard coded
	VideoWriter vw(argv[2], CV_FOURCC('M','P','E','G'), 25, Size(width*2, height*2));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}
    videoReader.Run();
	faceDetector.Run();
	/* Initial buffering for 10 frames */
	cout << "Buffering....." << endl;
	bool bufferDone = false;
	const int initialBufferSize = 5;
	/*
        The output image contains four parts
        +-------+-------+
        |       |       |
        |   1   |   2   |
        |       |       |
        +-------+-------+
        |       |       |
        |   3   |   4   |
        |       |       |
        +-------+-------+
        1 - Original input frame
        2 - Face detection result
        3 - Text localization result
        4 - Binary image of the text localization
    */
	Mat outputFrame(height*2, width*2, CV_8UC3);
	Mat inputFrame(height, width, CV_8UC3);
	Mat faceFrame(height, width, CV_8UC3);
	Mat textFrame(height, width, CV_8UC3);
	Mat binaryFrame(height, width, CV_8UC3);
	namedWindow("Result");
	do {

        faceDetector.GetFrame(faceFrame);
        imshow("Result", faceFrame);
        //vw << outputFrame;
	} while (waitKey(40) < 0);
	faceDetector.Stop();
    cout << "Done!" << endl;
	return 0;
}
