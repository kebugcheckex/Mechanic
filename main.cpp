/*
    Video Content Analysis

    This file is part of PWICE project.

    Contributors:
    - Wayne Hung
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

string PadZeroBefore(string str, int width)
{
    if (str.size() >= width) return str;
    for (int i = width - str.size(); i > 0; i--)
    {
        str = "0" + str;
    }
    return str;
}

string GetDateTime()
{
    time_t t = time(0);
    struct tm *now = localtime(&t);
    stringstream ss;
    ss << now->tm_year + 1900;
    string month(std::to_string(now->tm_mon + 1));
    ss << PadZeroBefore(month, 2);
    string day(std::to_string(now->tm_mday));
    ss << PadZeroBefore(day, 2);
    string hour(std::to_string(now->tm_hour));
    ss << PadZeroBefore(hour, 2);
    string minute(std::to_string(now->tm_min));
    ss << PadZeroBefore(minute, 2);
    string second(std::to_string(now->tm_sec));
    ss << PadZeroBefore(second, 2);
    return ss.str();
}

void DisplayText(Mat& frame, vector<string> lines)
{
    /*
     *  If the recognized text is longer than 12 characters, we need to wrap it
     *  around so that it will not overflow the display area.
     */
    vector<string> wrapped_lines;
    for (int i = 0; i < lines.size(); i++)
    {
        if (lines[i].size() <= 12) wrapped_lines.push_back(lines[i]);
        else
        {
            int j;
            for (j = 0; j < lines[i].size() / 12; j++)
            {
                wrapped_lines.push_back(lines[i].substr(j*12, 12));
            }
            wrapped_lines.push_back(lines[i].substr(j*12, lines[i].size() - j*12));
        }
    }
    /*
     *  TODO - Currently, the text display area is hard-coded. In the case that the recognized text
     *  is too long to fit in the box, we need to adjust the font size adaptively. This has not
     *  been implemented yet.
     */
    int width = frame.cols;
    int height = frame.rows;
    int x = width * 3 / 4 - height / 20;
    int y = height / 20;
    Mat wind = Mat::zeros(height/4, width/4, CV_8UC3);
    CvFont font = cvFontQt("Arial", 10, CV_RGB(255, 255, 255));
    for (int i = 0; i < wrapped_lines.size(); i++)
    {
        Point coord(0, 10 * (i + 1));
        addText(wind, wrapped_lines[i], coord, font);
    }
    wind.copyTo(frame(Rect(x, y, wind.cols, wind.rows)));
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " input-file" << endl;
        cout << "Input can be either a file on the disk or a URL to a video stream." << endl;
        return 1;
    }

    string xmlPath("./haarcascade_frontalface_default.xml");
    string inputFileName(argv[1]);
    VideoReader videoReader(inputFileName);
    FaceDetector faceDetector(&videoReader, xmlPath);
    TextDetector textDetector(&videoReader);
    int width = videoReader.GetSize().width;
	int height = videoReader.GetSize().height;
	string outputFileName(GetDateTime());
	outputFileName += ".mpg";
	cout << outputFileName << endl;

	VideoWriter vw(outputFileName, CV_FOURCC('M','P','E','G'), 25, Size(width, height));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}

	Mat outputFrame(height, width, CV_8UC3);
	Mat inputFrame(height, width, CV_8UC3);
    CvFont prompt_font = cvFontQt("Arial", 12.0, CV_RGB(255, 0, 0));
    vector<int> high_count;
    vector<Mat> theVideo;
    videoReader.Run();
    this_thread::sleep_for(chrono::milliseconds(100));
    textDetector.Run();
    faceDetector.Run();
	namedWindow("Result");

	bool user_quit = false;
	bool face_on = true;
	bool text_on = true;
	int content;
	int prompt_count = 0;
	string prompt_text = "";
	int display_delay = 0;
	vector<string> lines;
	while (!user_quit)
	{
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
                    for (int i = 0; i < tr[0].mask.rows; i++)
                    {
                        for (int j = 0; j < tr[0].mask.cols; j++)
                        {
                            auto pixel = tr[0].mask.at<Vec3b>(i, j);
                            if (pixel[1] != 0)
                            {
                                Vec3b color{0, 255, 0};
                                outputFrame.at<Vec3b>(i, j) = color;
                            }
                        }
                    }

                    for (auto it = tr.begin(); it != tr.end(); it++)
                    {
                        //rectangle(outputFrame, it->box, Scalar(0, 255, 255), 2);
                        if (display_delay == 0)
                        {
                            lines.push_back(it->text);
                            display_delay = 15;
                            content++;
                            int line = 0;

                        }

                    }
                }
            }

            DisplayText(outputFrame, lines);
            if (display_delay > 0) display_delay--;
            if (display_delay == 0) lines.clear();
        }

        if (prompt_text.size() > 0)
        {
            prompt_count++;
            if (prompt_count < 40)
            {
                addText(outputFrame, prompt_text, Point(5, 20), prompt_font);
            }
            else
            {
                prompt_text = "";
                prompt_count = 0;
            }
        }

        Mat result(width*2, height*2, CV_8UC3);
        resize(outputFrame, result, Size(width*2, height*2));
        imshow("Result", result);
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
            if (face_on)
            {
                face_on = false;
                prompt_text = "Face detection is turned off";
            }
            else
            {
                face_on = true;
                prompt_text = "Face detection is turned on";
            }
            break;
        case 't':
            if (text_on)
            {
                text_on = false;
                prompt_text = "Text detection is turned off";
            }
            else
            {
                text_on = true;
                prompt_text = "Text detection is turned on";
            }
            break;
        }
	}
	faceDetector.Stop();
	textDetector.Stop();
	videoReader.Stop();

	cout << "Processing highlight..." << endl;
	VideoCapture vc(outputFileName);
	if (!vc.isOpened())
	{
        cout << "Failed to open the recorded video!" << endl;
        return 8;
	}
	VideoWriter highvw(outputFileName + "_h.mpg", CV_FOURCC('M','P','E','G'), 25, Size(width, height));
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
            for (int j = i - 30; j < i + 30; j++)
            {
                highvw << theVideo[j];
            }
            i += 60;
        }
	}
	cout << "Total frames: " << theVideo.size() << endl;
	return 0;
}
