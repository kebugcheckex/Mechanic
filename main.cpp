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
    string xmlPath("/home/xinyu/projects/Mechanic/haarcascade_frontalface_default.xml");
    string inputFileName(argv[1]);
    VideoReader videoReader(inputFileName);
    FaceDetector faceDetector(&videoReader, xmlPath);
    TextDetector textDetector(&videoReader);
    int width = videoReader.GetSize().width;
	int height = videoReader.GetSize().height;
	string outputFileName(argv[2]);
	// TODO - FOURCC should not be hard coded
	VideoWriter vw(argv[2], CV_FOURCC('M','P','E','G'), 25, Size(width, height));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}


	Mat outputFrame(height, width, CV_8UC3);
	Mat inputFrame(height, width, CV_8UC3);
    CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(0, 255, 0) );
    vector<int> high_count;
    vector<Mat> theVideo;
    videoReader.Run();
    this_thread::sleep_for(chrono::milliseconds(100));
    textDetector.Run();
    faceDetector.Run();
	namedWindow("Result");
	bool user_quit = false;
	bool face_on = false;
	bool text_on = false;
	int content;
	while (!user_quit) {
        if (!videoReader.GetFrame(inputFrame))
            continue;
        content = 0;
        inputFrame.copyTo(outputFrame);
        if (face_on)
        {
            FaceResult fr;
            if (faceDetector.GetResult(fr))
            {
                if (fr.size() > 0)
                {
                    content++;
                    for (FaceResult::iterator it = fr.begin(); it != fr.end(); it++)
                    {
                        rectangle(outputFrame, *it, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        if (text_on)
        {
            TextResult tr;
            if (textDetector.GetResults(tr))
            {
                if (tr.size() > 0)
                {
                    content++;
                    for (TextResult::iterator it = tr.begin(); it != tr.end(); it++)
                    {
                        rectangle(outputFrame, it->box, Scalar(0, 255, 255), 2);
                        Point coord = Point(it->box.x - 15, it->box.y);
                        addText(outputFrame, it->text, coord, font);
                    }
                }

            }
        }
        imshow("Result", outputFrame);
        vw << outputFrame;
        high_count.push_back(content);
        Mat output = outputFrame.clone();
        theVideo.push_back(output);
        int key = waitKey(20);
        switch (key) {
        case 'q':
            user_quit = true;
            break;
        case 'f':
            face_on = !face_on;
            break;
        case 't':
            text_on = !text_on;
            break;
        }
	}
	faceDetector.Stop();
	textDetector.Stop();
	videoReader.Stop();
	cout << "Processing highlight..." << endl;
	VideoCapture vc(argv[2]);
	if (!vc.isOpened())
	{
        cout << "Failed to open the recorded video!" << endl;
        return 8;
	}
	VideoWriter highvw("highlight.mpg", CV_FOURCC('M','P','E','G'), 25, Size(width, height));
	if (!highvw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}
	if (theVideo.size() < 150)
	{
        cout << "Video too short!" << endl;
        return 11;
	}
	for (int i = 30; i < high_count.size() - 30; i++)
	{
        if (high_count[i] > 0)
        {
            //cout << "Got key frame: " << i << endl;
            for (int j = i - 30; j < i + 30; j++)
            {
                //cout << "Writing frame :" << j << endl;
                highvw << theVideo[j];
            }
            i += 60;
        }
	}
	cout << "Total frames: " << theVideo.size() << endl;
    cout << "Done!" << endl;
	return 0;
}
