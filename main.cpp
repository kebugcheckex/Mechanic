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
#include "VideoReader.h"
#include "FaceDetector.h"
#include "TextDetector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)   // We need two parameters, one for input file, and the other one for output file.
    {
        cout << "Usage: " << argv[0] << " input-file output-file" << endl;
        cout << "Input can be either a file on the disk or a URL to a video stream." << endl;
        return 1;
    }
    string xmlPath("haarcascade_frontalface_default.xml");
    string inputFileName(argv[1]);
    VideoReader videoReader(inputFileName);
    FaceDetector faceDetector(&videoReader, xmlPath);
    TextDetector textDetector(&videoReader);
    int width = videoReader.GetSize().width;
	int height = videoReader.GetSize().height;
	string outputFileName(argv[2]);
	// TODO - FOURCC should not be hard coded
	VideoWriter vw(argv[2], CV_FOURCC('M','P','E','G'), 25, Size(width*2, height*2));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}


	Mat outputFrame(height*2, width*2, CV_8UC3);
	Mat inputFrame(height, width, CV_8UC3);
    Mat faceResult(height, width, CV_8UC3);
    Mat textResult(height, width, CV_8UC3);
    Mat textBinary(height, width, CV_8UC3);
    textBinary = Scalar(0, 0, 0);

    CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(0, 255, 0) );

    videoReader.Run();
    textDetector.Run();
    faceDetector.Run();
	namedWindow("Result");
	do {
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
        videoReader.GetFrame(inputFrame);
        inputFrame.copyTo(outputFrame(Rect(0, 0, width, height)));

        faceResult = inputFrame.clone();
        FaceResult fr;
        if (faceDetector.GetResult(fr))
        {
            for (FaceResult::iterator it = fr.begin(); it != fr.end(); it++)
            {
                rectangle(faceResult, *it, Scalar(0, 255, 0), 2);
            }
        }
        faceResult.copyTo(outputFrame(Rect(width, 0, width, height)));


        textResult = inputFrame.clone();
        TextResult tr;
        if (textDetector.GetResults(tr))
        {
            for (TextResult::iterator it = tr.begin(); it != tr.end(); it++)
            {
                rectangle(textResult, it->box, Scalar(0, 0, 255), 2);
                Point coord = Point(it->box.x - 15, it->box.y);
                addText(textResult, it->text, coord, font);
            }
        }
        textResult.copyTo(outputFrame(Rect(0, height, width, height)));
        textBinary.copyTo(outputFrame(Rect(width, height, width, height)));
        imshow("Result", outputFrame);
        vw << outputFrame;
	} while (waitKey(40) < 0);
	faceDetector.Stop();
	textDetector.Stop();
	videoReader.Stop();
    cout << "Done!" << endl;
	return 0;
}
