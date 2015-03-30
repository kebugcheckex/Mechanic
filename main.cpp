/*
    Video Content Analysis

    This file is part of PWICE project.

    Contributors:
    - Wayne Huang
    - Xu Qiu
    - Brian Lan
    - Xinyu Chen
*/

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

    string inputFileName(argv[1]);
    VideoCapture vc;
    if (inputFileName == "0")   // If the input is string "0", we neet to convert it to number 0
        vc = VideoCapture(0);
    else
        vc = VideoCapture(inputFileName);

	if (!vc.isOpened())
	{
        cerr << "Error: Failed to open the input file " << inputFileName << endl;
        return 2;
	}
	namedWindow("Result");
	Mat inputFrame;
	vc >> inputFrame;
	int width = inputFrame.cols;
	int height = inputFrame.rows;

	string outputFileName(argv[2]);

	// TODO - FOURCC should not be hard coded
	VideoWriter vw(argv[2], CV_FOURCC('M','P','E','G'), 25, Size(width*2, height*2));
	if (!vw.isOpened())
	{
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
	}

	// TODO check whether the xml file exists
	CascadeClassifier face_cascade = CascadeClassifier("/home/xinyu/Videos/haarcascade_frontalface_default.xml");

	Mat outputFrame(height*2, width*2, CV_8UC3);
	cout << "Debug: width=" << width << "\theight=" << height << endl;
    Mat faceResult(height, width, CV_8UC3);
    Mat textResult(height, width, CV_8UC3);
    Mat textBinary(height, width, CV_8UC3);
    textBinary = Scalar(0, 0, 0);
	int total_frames = 0;
	int frame_count = 0;
	do {
        //if (frame_count % 5 == 0)
        {
            DetectFace(inputFrame, faceResult, face_cascade);
            DetectText(inputFrame, textResult, textBinary);
        }
        frame_count++;
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
        vw << outputFrame;
        if (waitKey(20) >= 0) break;
	} while (vc.read(inputFrame));
	cout << "Reach the end of the input file" << endl;
    cout << "Total frames = " << total_frames << endl;
    cout << "Done!" << endl;
	return 0;
}
