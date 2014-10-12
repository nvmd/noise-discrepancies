#include <vector>
#include <opencv2/core/core.hpp>

struct slic_segmentation_env_t {
    int    superpixel_count = 200;  //Desired number of superpixels.
    double compactness      = 20;   //Compactness factor
    // use a value ranging from 10 to 40
    // depending on your needs. Default is 10
};

void do_slic(const cv::Mat& bgr_input_image, const slic_segmentation_env_t& env,
        std::vector<cv::Mat>& cluster_masks);
