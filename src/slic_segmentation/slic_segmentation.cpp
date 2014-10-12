#include "slic_segmentation.hpp"

#include <vector>
#include <iostream>
#include <cassert>

#include <opencv2/core/core.hpp>

#include "slic/SLIC.h"
#include "noise_discrepancies/utils.hpp"

void do_slic(const cv::Mat& bgr_input_image, const slic_segmentation_env_t& env,
             std::vector<cv::Mat>& cluster_masks)
{
    //----------------------------------
    // Initialize parameters
    //----------------------------------

    assert(env.superpixel_count < bgr_input_image.cols * bgr_input_image.rows);

    // unsigned int (32 bits) to hold a pixel in ARGB format as follows:
    // from left to right,
    // the first 8 bits are for the alpha channel (and are ignored)
    // the next 8 bits are for the red channel
    // the next 8 bits are for the green channel
    // the last 8 bits are for the blue channel

    cv::Mat argb_image(bgr_input_image.size(), CV_8UC4);
    argb_image.setTo(cv::Scalar(255 /*alpha*/, 0 /*red*/, 0 /*green*/, 0 /*blue*/));
    int bgr_to_argb[] = { 0,3, // blue
            1,2, // green
            2,1 // red
    };
    cv::mixChannels(&bgr_input_image, 1, &argb_image, 1, bgr_to_argb, 3);

    std::cout << "ARGB Image: width=" << argb_image.cols << ", height=" << argb_image.rows
            << ", channels=" << argb_image.channels() << ", total=" << argb_image.total()
            << std::endl;
    assert(argb_image.isContinuous());

    int* cluster_labels = new int[argb_image.cols * argb_image.rows];
    int  cluster_count = 0;

    //----------------------------------
    // Perform SLIC on the input_image buffer
    //----------------------------------
    SLIC segment;

    //for a given number K of superpixels
    segment.PerformSLICO_ForGivenK((unsigned int*) argb_image.ptr(),
            argb_image.cols, argb_image.rows,
            cluster_labels, cluster_count,
            env.superpixel_count, env.compactness);
    //for a given grid step size / desired superpixel size
    //segment.PerformSLICO_ForGivenStepSize(img, argb_image.cols, argb_image.rows,
    // cluster_labels, cluster_count, stepsize, compactness);

    //----------------------------------
    // Generate labels matrix and segment masks
    //----------------------------------
    cv::Mat labels(argb_image.size(), CV_32SC1);
    std::copy(cluster_labels, cluster_labels + argb_image.cols * argb_image.rows,
            (unsigned int*) labels.ptr());

    cluster_masks.clear();
    for (int i = 0; i < cluster_count; ++i) {
        cluster_masks.push_back(cv::Mat::zeros(argb_image.size(), CV_8UC1));
    }
    generate_cluster_masks(argb_image, labels, cluster_masks);

    //----------------------------------
    // Save the labels to a text file
    //----------------------------------
//	segment.SaveSuperpixelLabels(cluster_labels, argb_image.cols, argb_image.rows, filename, savepath);

    //----------------------------------
    // Draw boundaries around segments
    //----------------------------------
    //for black contours around superpixels
//	segment.DrawContoursAroundSegments((unsigned int*) argb_image.ptr(),
//                                       cluster_labels,
//                                       argb_image.cols, argb_image.rows,
//                                       0xff000000);
    segment.DrawContoursAroundSegmentsTwoColors((unsigned int*) argb_image.ptr(),
            cluster_labels,
            argb_image.cols, argb_image.rows);

    //----------------------------------
    // Save the input_image with segment boundaries.
    //----------------------------------
    write_argb_image(argb_image, std::string("slic_argb_output.jpg"));

    //----------------------------------
    // Clean up
    //----------------------------------
    if (cluster_labels) {
        delete[] cluster_labels;
    }
}
