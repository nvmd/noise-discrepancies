#include <string>
#include <iostream>
#include <cmath>

// #include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>	//imread, imwrite
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

#include "SLIC.h"

#define OPENCV_SLIC

bool write_argb_image(const cv::Mat& argb_image,
                      const std::string& output_filename)
{
    cv::Mat output_image(argb_image.size(), CV_8UC3);
    int argb_to_bgr[] = { 3,0, // blue
                          2,1, // green
                          1,2 // red
                        };
    cv::mixChannels(&argb_image, 1, &output_image, 1, argb_to_bgr, 3);
    std::cout << "Output Image: name=" << output_filename
            << ", width=" << output_image.cols << ", height=" << output_image.rows
            << ", channels=" << output_image.channels() << ", total=" << output_image.total()
            << std::endl;
    return cv::imwrite(output_filename, output_image);
}

void generate_cluster_masks(cv::Mat &image, const cv::Mat &labels,
                            std::vector<cv::Mat> &cluster_masks)
{
    for(int y = 0; y < image.rows; ++y) {
        for(int x = 0; x < image.cols; ++x) {
            int cluster_label = labels.at<int>(y, x);
            cluster_masks[cluster_label].at<uchar>(y, x) = 255;
        }
    }
}

void draw_contour_for_masked_segment(cv::Mat &segment_mask,
                                     cv::Mat &segmented_contour_image,
                                     const cv::Scalar& color)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(segment_mask, contours,
                     CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); ++i) {
        cv::drawContours(segmented_contour_image, contours, i, color, 2);
    }
}

int main(int argc, char** argv)
{
    const std::string input_filename("input0.png");
    const std::string output_filename("output.jpg");
    const std::string segmented_contour_image_filename("contours.jpg");

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

    std::cout << "ARGB Image: width=" << argb_image.cols << ", height=" << argb_image.rows
              << ", channels=" << argb_image.channels() << ", total=" << argb_image.total()
              << std::endl;
    assert(argb_image.isContinuous());

	int width = argb_image.cols;
    int height = argb_image.rows;
	// unsigned int (32 bits) to hold a pixel in ARGB format as follows:
	// from left to right,
	// the first 8 bits are for the alpha channel (and are ignored)
	// the next 8 bits are for the red channel
	// the next 8 bits are for the green channel
	// the last 8 bits are for the blue channel
    unsigned int* pbuff = (unsigned int*) argb_image.ptr();

	//----------------------------------
	// Initialize parameters
	//----------------------------------
    //Desired number of superpixels.
	const int    superpixel_count = 200;
    //Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	const double compactness      = 20;

    assert(superpixel_count < width * height);

	int* cluster_labels = new int[width * height];
	int  cluster_count = 0;
//    std::string filename = "yourfilename.jpg";
//    std::string savepath = "yourpathname";
	//----------------------------------
	// Perform SLIC on the input_image buffer
	//----------------------------------
	SLIC segment;

    //for a given number K of superpixels
	segment.PerformSLICO_ForGivenK(pbuff, width, height,
                                   cluster_labels, cluster_count,
                                   superpixel_count, compactness);
    //for a given grid step size / desired superpixel size
    //segment.PerformSLICO_ForGivenStepSize(img, width, height, labels, cluster_count, stepsize, compactness);

    //----------------------------------
    // Generate labels matrix and segment masks
    //----------------------------------
#if defined(OPENCV_SLIC)
    cv::Mat labels(argb_image.size(), CV_32SC1);
    std::copy(cluster_labels, cluster_labels + width * height,
              (unsigned int*) labels.ptr());

    std::vector<cv::Mat> cluster_masks;
    for (int i = 0; i < cluster_count; ++i) {
        cluster_masks.push_back(cv::Mat::zeros(argb_image.size(), CV_8UC1));
    }
    generate_cluster_masks(argb_image, labels, cluster_masks);
#endif

    //----------------------------------
	// Save the labels to a text file
	//----------------------------------
//	segment.SaveSuperpixelLabels(cluster_labels, width, height, filename, savepath);
	//----------------------------------
	// Draw boundaries around segments
	//----------------------------------
#if !defined(OPENCV_SLIC)
    //for black contours around superpixels
//	segment.DrawContoursAroundSegments(pbuff, cluster_labels, width, height, 0xff000000);
    segment.DrawContoursAroundSegmentsTwoColors(pbuff, cluster_labels,
                                                width, height);
#else
    // Draw contour for each segment
    cv::Mat segmented_contour_image;
    argb_image.copyTo(segmented_contour_image);

    cv::RNG rng(890);
    for (size_t i = 0; i < cluster_masks.size(); ++i) {
        // "ensure" that different segments are marked with different colors
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                      rng.uniform(0, 255),
                                      rng.uniform(0, 255));
        draw_contour_for_masked_segment(cluster_masks[i],
                                        segmented_contour_image,
                                        color);
    }
#endif

	//----------------------------------
	// Save the input_image with segment boundaries.
	//----------------------------------
    std::copy(pbuff, pbuff + width * height,
              (unsigned int*) argb_image.ptr());
    write_argb_image(argb_image, output_filename);

    write_argb_image(segmented_contour_image,
                     segmented_contour_image_filename);

	//----------------------------------
	// Clean up
	//----------------------------------
    if (cluster_labels) {
        delete[] cluster_labels;
    }

	return 0;
}
