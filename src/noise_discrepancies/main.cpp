#include <string>
#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

#include "SLIC.h"

bool write_bgr_image(const cv::Mat& output_image,
                     const std::string& output_filename)
{
    std::cout << "Output Image: name=" << output_filename
            << ", width=" << output_image.cols << ", height=" << output_image.rows
            << ", channels=" << output_image.channels() << ", total=" << output_image.total()
            << std::endl;
    return cv::imwrite(output_filename, output_image);
}

bool write_argb_image(const cv::Mat& argb_image,
                      const std::string& output_filename)
{
    cv::Mat output_image(argb_image.size(), CV_8UC3);
    int argb_to_bgr[] = { 3,0, // blue
                          2,1, // green
                          1,2 // red
                        };
    cv::mixChannels(&argb_image, 1, &output_image, 1, argb_to_bgr, 3);
    return write_bgr_image(output_image, output_filename);
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
                                     const cv::Scalar& color,
                                     const int thickness = 2)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(segment_mask, contours,
                     CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); ++i) {
        cv::drawContours(segmented_contour_image, contours, i, color, thickness);
    }
}

void do_slic(const cv::Mat& bgr_input_image,
             const int superpixel_count, const double compactness,
             std::vector<cv::Mat>& cluster_masks)
{
    //----------------------------------
    // Initialize parameters
    //----------------------------------

    assert(superpixel_count < bgr_input_image.cols * bgr_input_image.rows);

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
                                   superpixel_count, compactness);
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

int main(int argc, char** argv)
{
    const std::string input_filename(argv[1]);
    const std::string segmented_contour_image_filename("contours.jpg");
    const int    slic_superpixel_count = 200;    //Desired number of superpixels.
    const double slic_compactness = 20;     //Compactness factor
                                            // use a value ranging from 10 to 40
                                            // depending on your needs. Default is 10
    const int median_filter_kernel_size = 3;
    const cv::Size gaussian_filter_kernel_size(3, 3);
    double gaussian_filter_kernel_sigma_x = 0;

    cv::Mat input_image = cv::imread(input_filename);
    if (input_image.data == nullptr) {
        std::cout << "Can't read input image" << std::endl;
        return -1;
    }

    std::cout << "Input Image: name=" << input_filename
                  << ", width=" << input_image.cols << ", height=" << input_image.rows
                  << ", channels=" << input_image.channels() << ", total=" << input_image.total()
                  << std::endl;

	// Do SLIC (superpixel) segmentation
    std::vector<cv::Mat> cluster_masks;
    do_slic(input_image, slic_superpixel_count, slic_compactness, cluster_masks);

    // Draw contour for each segment and save the image with segment boundaries.
    cv::Mat segmented_contour_image;
    input_image.copyTo(segmented_contour_image);
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
    write_bgr_image(segmented_contour_image,
                    segmented_contour_image_filename);

    // Estimated noise for the input image
    std::vector<cv::Mat> noise_estimation;
    cv::Mat median_filtered;
    cv::medianBlur(input_image, median_filtered, median_filter_kernel_size);
    noise_estimation.push_back(input_image - median_filtered);
    write_bgr_image(median_filtered, "median_noise_filtered.jpg");
    write_bgr_image(noise_estimation.back(), "median_noise_est.jpg");
    cv::Mat gaussian_filtered;
    cv::GaussianBlur(input_image, gaussian_filtered,
                     gaussian_filter_kernel_size, gaussian_filter_kernel_sigma_x);
    noise_estimation.push_back(input_image - gaussian_filtered);
    write_bgr_image(gaussian_filtered, "gaussian_noise_filtered.jpg");
    write_bgr_image(noise_estimation.back(), "gaussian_noise_est.jpg");



    cv::Mat3d segment_features(cluster_masks.size(),
                               2 * noise_estimation.size());
    for (size_t i = 0; i < cluster_masks.size(); ++i) {
        for (size_t j = 0; j < noise_estimation.size(); ++j) {
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::meanStdDev(noise_estimation[j],
                           mean, stddev,
                           cluster_masks[i]);

            // F^cd_s:
            // {(mean_d1c1, mean_d1c2, mean_d1c3),
            //  (stddev_d1c1, stddev_d1c2, stddev_d1c3)}

            cv::Vec3d& filter_mean   = segment_features.at<cv::Vec3d>(i, j * 2);
            cv::Vec3d& filter_stddev = segment_features.at<cv::Vec3d>(i, j * 2 + 1);
            filter_mean[0]   = mean.val[0];
            filter_mean[1]   = mean.val[1];
            filter_mean[2]   = mean.val[2];
            filter_stddev[0] = stddev.val[0];
            filter_stddev[1] = stddev.val[1];
            filter_stddev[2] = stddev.val[2];
        }
    }

    cv::Mat1d segment_feature_norms(segment_features.rows, 1);
    for (size_t i = 0; i < segment_features.rows; ++i) {
        double norm = cv::norm(segment_features.row(i), cv::NORM_L2);
        segment_feature_norms.at<double>(i, 0) = norm;
    }

    cv::Scalar norm_mean_scalar = cv::mean(segment_feature_norms);
    double norm_mean = norm_mean_scalar.val[0];
    double norm_max;
    cv::minMaxIdx(segment_feature_norms, nullptr, &norm_max,
                                         nullptr, nullptr);

    std::cout << "Features: segments_count=" << segment_features.rows
              << ", components=" << segment_features.cols * segment_features.channels()
              << ", filter_components=" << segment_features.cols
              << ", avg_norm=" << norm_mean
              << ", max_norm=" << norm_max
              << std::endl;

    cv::Mat1d segment_feature_deviations(segment_feature_norms.rows, 1);
    for (size_t i = 0; i < segment_feature_norms.rows; ++i) {
        double norm = segment_feature_norms.at<double>(i, 0);
        segment_feature_deviations.at<double>(i, 0) = std::abs(norm_mean - norm);
    }

    cv::normalize(segment_feature_deviations, segment_feature_deviations,
                  0, 255, cv::NORM_MINMAX);

    cv::Mat noise_discrepancies;
    input_image.copyTo(noise_discrepancies);
    for (size_t i = 0; i < cluster_masks.size(); ++i) {
        cv::Scalar color = cv::Scalar(segment_feature_deviations.at<double>(i, 0),
                                      0, 0);
        draw_contour_for_masked_segment(cluster_masks[i],
                                        noise_discrepancies,
                                        color, CV_FILLED);
    }
    write_bgr_image(noise_discrepancies, "noise_discrepancies_map.jpg");

	return 0;
}
