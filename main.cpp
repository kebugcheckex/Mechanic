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
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"
#include "utils.h"

#define NUM_FEATURES 11
#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

using namespace std;
using namespace cv;
using namespace tesseract;

bool DetectFace(Mat& inputImage, Mat& outputImage, CascadeClassifier& classifier)
{
    cout << "Debug checkpoint: Detect Face" << endl;
	vector<Rect> objects;
	bool detected = false;
	inputImage.copyTo(outputImage);
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
    cout << "Debug checkpoint: Detect Text" << endl;
    inputImage.copyTo(outputImage);
    tesseract::TessBaseAPI api;
    api.Init(NULL, "eng");
    vector<Region> regions;

    ::MSER mser8(false,25,0.000008,0.03,1,0.7);

    RegionClassifier region_boost("boost_train/trained_boost_char.xml", 0);
    GroupClassifier  group_boost("boost_train/trained_boost_groups.xml", &region_boost);
    Mat grey, lab_img, gradient_magnitude, segmentation, all_segmentations;
    cvtColor(inputImage, grey, CV_BGR2GRAY);
    cvtColor(inputImage, lab_img, CV_BGR2Lab);

    gradient_magnitude = Mat_<double>(inputImage.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    segmentation = Mat::zeros(inputImage.size(),CV_8UC3);
    all_segmentations = Mat::zeros(240,320*11,CV_8UC3);

    int dims[NUM_FEATURES] = {3,3,3,3,3,3,3,3,3,5,5};

    for (int step =1; step<3; step++)
    {
        if (step == 2)
        grey = 255-grey;
            //double t_tot = (double)cvGetTickCount();
            //double t = (double)cvGetTickCount();
        mser8((uchar*)grey.data, grey.cols, grey.rows, regions);
            //t = cvGetTickCount() - t;
            //cout << "Detected " << regions.size() << " regions" << " in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
            //t = (double)cvGetTickCount();
        for (int i=0; i<regions.size(); i++)
            regions[i].er_fill(grey);
            //t = cvGetTickCount() - t;
            //cout << "Regions filled in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
            //t = (double)cvGetTickCount();
        double max_stroke = 0;
        for (int i=regions.size()-1; i>=0; i--)
        {
            regions[i].extract_features(lab_img, grey, gradient_magnitude);
            if ( (regions.at(i).stroke_std_/regions.at(i).stroke_mean_ > 0.8) || (regions.at(i).num_holes_>2) || (regions.at(i).bbox_.width <=3) || (regions.at(i).bbox_.height <=3) )
                regions.erase(regions.begin()+i);
            else
                max_stroke = max(max_stroke, regions[i].stroke_mean_);
        }
            //t = cvGetTickCount() - t;
            //cout << "Features extracted in " << t/((double)cvGetTickFrequency()*1000.) << " ms." << endl;
            //t = (double)cvGetTickCount();
        MaxMeaningfulClustering 	mm_clustering(METHOD_METR_SINGLE, METRIC_SEUCLIDEAN);

        vector< vector<int> > meaningful_clusters;
        vector< vector<int> > final_clusters;
        Mat co_occurrence_matrix = Mat::zeros((int)regions.size(), (int)regions.size(), CV_64F);

        for (int f=0; f<NUM_FEATURES; f++)
        {
            unsigned int N = regions.size();
            if (N<3) break;
            int dim = dims[f];
            t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));
            int count = 0;
            for (int i=0; i<regions.size(); i++)
            {
                data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/inputImage.cols;
                data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/inputImage.rows;
				switch (f)
				{
				case 0:
				    data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
				    break;
				case 1:
				    data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;
				    break;
				case 2:
				    data[count+2] = (t_float)regions.at(i).bbox_.y/inputImage.rows;
				    break;
				case 3:
				    data[count+2] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height)/inputImage.rows;
				    break;
				case 4:
				    data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(inputImage.rows,inputImage.cols);
				    break;
				case 5:
				    data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;
				    break;
				case 6:
				    data[count+2] = (t_float)regions.at(i).area_/(inputImage.rows*inputImage.cols);
				    break;
				case 7:
				    data[count+2] = (t_float)(regions.at(i).bbox_.height*regions.at(i).bbox_.width)/(inputImage.rows*inputImage.cols);
				    break;
				case 8:
				    data[count+2] = (t_float)regions.at(i).gradient_mean_/255;
				    break;
				case 9:
				    data[count+2] = (t_float)regions.at(i).color_mean_.at(0)/255;
				    data[count+3] = (t_float)regions.at(i).color_mean_.at(1)/255;
				    data[count+4] = (t_float)regions.at(i).color_mean_.at(2)/255;
				    break;
				case 10:
				    data[count+2] = (t_float)regions.at(i).boundary_color_mean_.at(0)/255;
				    data[count+3] = (t_float)regions.at(i).boundary_color_mean_.at(1)/255;
				    data[count+4] = (t_float)regions.at(i).boundary_color_mean_.at(2)/255;
				    break;
                }
                count = count+dim;
            }
            mm_clustering(data, N, dim, METHOD_METR_SINGLE, METRIC_SEUCLIDEAN, &meaningful_clusters); // TODO try accumulating more evidence by using different methods and metrics
            for (int k=0; k<meaningful_clusters.size(); k++)
            {
                //if ( group_boost(&meaningful_clusters.at(k), &regions)) // TODO try is it's betetr to accumulate only the most probable text groups
                accumulate_evidence(&meaningful_clusters.at(k), 1, &co_occurrence_matrix);
                if ( (group_boost(&meaningful_clusters.at(k), &regions) >= DECISION_THRESHOLD_SF) )
                {
                    final_clusters.push_back(meaningful_clusters.at(k));
                }
            }

            Mat tmp_segmentation = Mat::zeros(inputImage.size(),CV_8UC3);
            Mat tmp_all_segmentations = Mat::zeros(240,320*11,CV_8UC3);
            drawClusters(tmp_segmentation, &regions, &meaningful_clusters);
            Mat tmp = Mat::zeros(240,320,CV_8UC3);
            resize(tmp_segmentation,tmp,tmp.size());
            tmp.copyTo(tmp_all_segmentations(Rect(320*f,0,320,240)));
            all_segmentations = all_segmentations + tmp_all_segmentations;

            free(data);
            meaningful_clusters.clear();
        }
        double minVal;
        double maxVal;
        minMaxLoc(co_occurrence_matrix, &minVal, &maxVal);
        maxVal = NUM_FEATURES - 1; //TODO this is true only if you are using "grow == 1" in accumulate_evidence function
        minVal=0;
        co_occurrence_matrix = maxVal - co_occurrence_matrix;
        co_occurrence_matrix = co_occurrence_matrix / maxVal;
        //we want a sparse matrix
        t_float *D = (t_float*)malloc((regions.size()*regions.size()) * sizeof(t_float));
        int pos = 0;
        for (int i = 0; i<co_occurrence_matrix.rows; i++)
        {
            for (int j = i+1; j<co_occurrence_matrix.cols; j++)
            {
                D[pos] = (t_float)co_occurrence_matrix.at<double>(i, j);
                pos++;
            }
        }
            // fast clustering from the co-occurrence matrix
        mm_clustering(D, regions.size(), METHOD_METR_AVERAGE, &meaningful_clusters); //  TODO try with METHOD_METR_COMPLETE
        free(D);
        for (int i=meaningful_clusters.size()-1; i>=0; i--)
        {
            //if ( (! group_boost(&meaningful_clusters.at(i), &regions)) || (meaningful_clusters.at(i).size()<3) )
            if ( (group_boost(&meaningful_clusters.at(i), &regions) >= DECISION_THRESHOLD_EA) )
            {
                final_clusters.push_back(meaningful_clusters.at(i));
            }
        }
        drawClusters(segmentation, &regions, &final_clusters);
        if (step == 2)
        {
            cvtColor(segmentation, grey, CV_BGR2GRAY);
            threshold(grey,grey,1,255,CV_THRESH_BINARY);
            if (countNonZero(grey) < inputImage.cols*inputImage.rows/2)
                threshold(grey,grey,1,255,THRESH_BINARY_INV);
                //imwrite("out.png", grey);*/
//            if (argc > 2)
//            {
//                Mat gt;
//                gt = imread(argv[2]);
//                cvtColor(gt, gt, CV_RGB2GRAY);
//                threshold(gt, gt, 1, 255, CV_THRESH_BINARY_INV); // <- for KAIST gt
//                    //threshold(gt, gt, 254, 255, CV_THRESH_BINARY); // <- for ICDAR gt
//                Mat tmp_mask = (255-gt) & (grey);
//                cout << "Pixel level recall = " << (float)countNonZero(tmp_mask) / countNonZero(255-gt) << endl;
//                cout << "Pixel level precission = " << (float)countNonZero(tmp_mask) / countNonZero(grey) << endl;
//            }
//            else
//            {
            cout << "using tesseract api" << endl;
            api.SetImage((uchar*) grey.data, grey.cols, grey.rows, 1, grey.cols);
            cout << "get some boxes" << endl;
            Boxa* boxes = api.GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
            if(boxes == NULL) continue;
            printf("Found %d textline image components.\n", boxes->n);
            for (int i = 0; i < boxes->n; i++)
            {
                BOX* box = boxaGetBox(boxes, i, L_CLONE);
                api.SetRectangle(box->x, box->y, box->w, box->h);
                char* ocrResult = api.GetUTF8Text();
                int conf = api.MeanTextConf();
                if (conf < 80) continue;
                fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
                        i, box->x, box->y, box->w, box->h, conf, ocrResult);
                rectangle(outputImage, Rect(box->x, box->y, box->w, box->h), Scalar(0, 255, 0), 2);
                CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(0, 255, 0) );
                Point coord = Point(box->x-15, box->y );
                addText(outputImage, ocrResult, coord, font );
            }
            cout << "Grey size " << grey.cols << " " << grey.rows << endl;
            cvtColor(grey, binaryImage, CV_GRAY2RGB);
        }
        regions.clear();
    }
    return true;
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
#endif
	return 0;
}
