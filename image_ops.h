/**
* @author - neil okikiolu
*/

#ifndef __ROS_IMAGE_OPS_
#define __ROS_IMAGE_OPS_

#include <opencv2/core/mat.hpp>

namespace imgops
{
    void saveImage(cv::Mat *cvImage, std::string *base_path, std::string *suffix);
}

#endif
