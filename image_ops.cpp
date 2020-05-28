/**
* @author - neil okikiolu
*/

#include "image_ops.h"

#include <iostream>
#include <ctime>
#include <sstream>
#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace imgops
{
    void saveImage(cv::Mat *cvImage, std::string *base_path, std::string *suffix) {
        // create date time string for captures
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H-%M-%S");
        auto str = oss.str();

        // combine date time string with path for final image path
        std::string out_string = *base_path + "capture_" + str + *suffix +".png";
        ROS_INFO (
            "Starting to save image: %s",
            out_string.c_str()
        );

        bool result = false;

        try
        {
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            // try saving the cv::Mat as a png
            result = imwrite(out_string, *cvImage, compression_params);
        }
        catch (const cv::Exception& ex)
        {
            ROS_ERROR(
                "Exception converting image to PNG format: %s\n",
                ex.what()
            );
        }

        if (result) {
            ROS_INFO(
                "Saved %s",
                out_string.c_str()
            );
        }
        else {
            ROS_ERROR("ERROR: Can't save PNG file.");
            return;
        }
    }
}
