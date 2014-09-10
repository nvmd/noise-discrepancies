#include <string>
#include <iostream>
#include <cmath>

// #include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>	//imread, imwrite
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

#include "SLIC.h"

int main(int argc, char** argv)
{
    std::string input_filename("input0.png");
    std::string output_filename("output.jpg");

    cv::Mat input_image = cv::imread(input_filename);
    if (input_image.data == nullptr) {
        std::cout << "Can't read input image" << std::endl;
        return -1;
    }

    std::cout << "Input Image: name=" << input_filename
                  << ", width=" << input_image.cols << ", height=" << input_image.rows
                  << ", channels=" << input_image.channels() << ", total=" << input_image.total()
                  << std::endl;

    cv::Mat argb_image(input_image.size(), CV_8UC4);
    argb_image.setTo(cv::Scalar(255 /*alpha*/, 0 /*red*/, 0 /*green*/, 0 /*blue*/));
    int bgr_to_argb[] = { 0,3, // blue
                          1,2, // green
                          2,1 // red
                        };
    cv::mixChannels(&input_image, 1, &argb_image, 1, bgr_to_argb, 3);

    assert(argb_image.isContinuous());
    std::cout << "ARGB Image: width=" << argb_image.cols << ", height=" << argb_image.rows
              << ", channels=" << argb_image.channels() << ", total=" << argb_image.total()
              << std::endl;

	int width = argb_image.cols;
    int height = argb_image.rows;
	// unsigned int (32 bits) to hold a pixel in ARGB format as follows:
	// from left to right,
	// the first 8 bits are for the alpha channel (and are ignored)
	// the next 8 bits are for the red channel
	// the next 8 bits are for the green channel
	// the last 8 bits are for the blue channel
    unsigned int* pbuff = (unsigned int*) argb_image.ptr();
//	ReadImage(pbuff, width, height);//YOUR own function to read an input_image into the ARGB format

	//----------------------------------
	// Initialize parameters
	//----------------------------------
    //Desired number of superpixels.
	int    superpixel_count = 200;
    //Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	double compactness      = 20;

	int* klabels = new int[width * height];
	int numlabels = 0;
//    std::string filename = "yourfilename.jpg";
//    std::string savepath = "yourpathname";
	//----------------------------------
	// Perform SLIC on the input_image buffer
	//----------------------------------
	SLIC segment;
	segment.PerformSLICO_ForGivenK(pbuff, width, height, klabels, numlabels, superpixel_count, compactness);
	// Alternately one can also use the function PerformSLICO_ForGivenStepSize() for a desired superpixel size
	//----------------------------------
	// Save the labels to a text file
	//----------------------------------
//	segment.SaveSuperpixelLabels(klabels, width, height, filename, savepath);
	//----------------------------------
	// Draw boundaries around segments
	//----------------------------------
//	segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff000000);
    segment.DrawContoursAroundSegmentsTwoColors(pbuff, klabels, width, height);
	//----------------------------------
	// Save the input_image with segment boundaries.
	//----------------------------------
//	SaveSegmentedImageFile(pbuff, width, height);//YOUR own function to save an ARGB buffer as an input_image
	//----------------------------------
	// Clean up
	//----------------------------------
//	if(pbuff) delete [] pbuff;
	if(klabels) delete [] klabels;

    std::copy(pbuff, pbuff + width * height, (unsigned int*) argb_image.ptr());

    cv::Mat output_image(argb_image.size(), CV_8UC3);
    int argb_to_bgr[] = { 3,0, // blue
            2,1, // green
            1,2 // red
    };
    cv::mixChannels(&argb_image, 1, &output_image, 1, argb_to_bgr, 3);
    cv::imwrite(output_filename, output_image);
    std::cout << "Output Image: name=" << output_filename
            << ", width=" << output_image.cols << ", height=" << output_image.rows
            << ", channels=" << output_image.channels() << ", total=" << output_image.total()
            << std::endl;

	return 0;
}
