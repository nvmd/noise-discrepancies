#include <string>
#include <iostream>
#include <cmath>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.hpp"
#include "slic_segmentation/slic_segmentation.hpp"

struct noise_estimation_env_t {
    int      median_filter_kernel_size      = 3;
    cv::Size gaussian_filter_kernel_size    = cv::Size(3, 3);
    double   gaussian_filter_kernel_sigma_x = 0;
    cv::Size average_filter_kernel_size     = cv::Size(3, 3);
};

void estimate_noise(const cv::Mat& input_image, std::vector<cv::Mat>& estimations,
        const noise_estimation_env_t& env)
{
    // "salt and pepper" noise
    cv::Mat median_filtered;
    cv::medianBlur(input_image, median_filtered, env.median_filter_kernel_size);
    estimations.push_back(input_image - median_filtered);
    write_bgr_image(median_filtered, "median_noise_filtered.jpg");
    write_bgr_image(estimations.back(), "median_noise_est.jpg");

    // high-frequency noise
    cv::Mat gaussian_filtered;
    cv::GaussianBlur(input_image, gaussian_filtered,
            env.gaussian_filter_kernel_size, env.gaussian_filter_kernel_sigma_x);
    estimations.push_back(input_image - gaussian_filtered);
    write_bgr_image(gaussian_filtered, "gaussian_noise_filtered.jpg");
    write_bgr_image(estimations.back(), "gaussian_noise_est.jpg");

    cv::Mat average_filtered;
    cv::blur(input_image, average_filtered, env.average_filter_kernel_size);
    estimations.push_back(input_image - average_filtered);
    write_bgr_image(average_filtered, "average_noise_filtered.jpg");
    write_bgr_image(estimations.back(), "average_noise_est.jpg");
}

int main(int argc, char** argv)
{
    const std::string input_filename(argv[1]);
    const std::string segmented_contour_image_filename("contours.jpg");
    slic_segmentation_env_t slic_segmentation_env;
    noise_estimation_env_t  noise_estimation_env;

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
    do_slic(input_image, slic_segmentation_env, cluster_masks);

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
    write_bgr_image(segmented_contour_image, segmented_contour_image_filename);

    // Estimated noise for the input image
    std::vector<cv::Mat> noise_estimation;
    estimate_noise(input_image, noise_estimation, noise_estimation_env);

    // Segment feature vectors
    cv::Mat3d segment_features(cluster_masks.size(),
                               2 * noise_estimation.size());
    for (size_t i = 0; i < cluster_masks.size(); ++i) {
        // Generate feature vector for segment
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
    cv::absdiff(segment_feature_norms, norm_mean, segment_feature_deviations);

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
