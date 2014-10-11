#ifndef _NOISE_DISCREPANCIES_UTILS_HPP_
#define _NOISE_DISCREPANCIES_UTILS_HPP_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

bool write_bgr_image(const cv::Mat& output_image, const std::string& output_filename);
bool write_argb_image(const cv::Mat& argb_image, const std::string& output_filename);

void generate_cluster_masks(cv::Mat &image, const cv::Mat &labels,
                            std::vector<cv::Mat> &cluster_masks);

void draw_contour_for_masked_segment(cv::Mat &segment_mask,
                                     cv::Mat &segmented_contour_image,
                                     const cv::Scalar& color,
                                     const int thickness = 2);

#endif  // _NOISE_DISCREPANCIES_UTILS_HPP_
