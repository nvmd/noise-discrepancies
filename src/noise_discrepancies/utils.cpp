#include <string>
#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool write_bgr_image(const cv::Mat& output_image, const std::string& output_filename)
{
    std::cout << "Output Image: name=" << output_filename
            << ", width=" << output_image.cols << ", height=" << output_image.rows
            << ", channels=" << output_image.channels() << ", total=" << output_image.total()
            << std::endl;
    return cv::imwrite(output_filename, output_image);
}

bool write_argb_image(const cv::Mat& argb_image, const std::string& output_filename)
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
