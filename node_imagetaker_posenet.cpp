
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/VisionInfo.h>

#include "poseNet.h"
#include <jetson-utils/cudaMappedMemory.h>

#include "image_converter.h"
#include "image_ops.h"

#include <unordered_map>

using namespace cv;
using namespace std;

// globals
poseNet* net = NULL;

imageConverter* in_cvt = NULL;

ros::Publisher* pose_pub = NULL;


// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr& input )
{
    ROS_INFO ("Received Image");
    
    // convert sensor_msgs[rgb] to opencv[brg]
    cv_bridge::CvImagePtr cv_ptr;
    cv_bridge::CvImagePtr cv_ptr_flip; // pointer for flipped image
    try
    {
        // sensor_msgs::image_encodings::BGR8
        cv_ptr = cv_bridge::toCvCopy(
            input,
            sensor_msgs::image_encodings::BGR8
        );

        cv_ptr_flip = cv_bridge::toCvCopy(
            input,
            sensor_msgs::image_encodings::BGR8
        );
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    // we are doing a 180 deg flip since
    // my camera is upside down
    const int img_flip_mode_ = -1;
    // flip the image
    cv::flip(cv_ptr->image, cv_ptr_flip->image, img_flip_mode_);
    
    // convert converted image back to a sensor_msgs::ImagePtr
    // for use with nvidia / other ML algorithms
    sensor_msgs::ImagePtr flippedImage = cv_ptr_flip->toImageMsg();

    // convert the image TO reside on GPU
    // the converting TO and converting FROM are the SAME funtion name
    // with different signatures
    if( !in_cvt || !in_cvt->Convert(flippedImage) )
    {
        ROS_ERROR (
            "failed to convert %ux%u %s image",
            flippedImage->width,
            flippedImage->height,
            flippedImage->encoding.c_str()
        );
        return;
    }
    else {
        ROS_INFO (
            "Converted %ux%u %s image",
            flippedImage->width,
            flippedImage->height,
            flippedImage->encoding.c_str()
        );
    }
    
    // generate pose parameters and overlay on the input image
    const bool processed = net->Process(
        in_cvt->ImageGPU(),
        in_cvt->GetWidth(),
        in_cvt->GetHeight()
    );
    
    // process the segmentation network
    if (!processed)
    {
        ROS_ERROR(
            "failed to process pose on %ux%u image",
            flippedImage->width,
            flippedImage->height
        );
        return;
    }

    CUDA(cudaDeviceSynchronize());

    // populate the message
    sensor_msgs::Image msg;
    // get our image with overlays back from the GPU
    // the converting TO and converting FROM are the SAME funtion name
    // with different signatures
    if( !in_cvt->Convert(msg, sensor_msgs::image_encodings::BGR8) ) {
        return;
    }
    pose_pub->publish(msg);
}



int main (int argc, char **argv) {
    ros::init(argc, argv, "imagetaker_posenet");
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
       
    // use this so we can pass parameters via command line e.g
    // rosrun jetsoncam imagetaker_homographynet 

    // create network using the built-in model
    /*
    net = poseNet::Create();
    */
    
    net = poseNet::Create(
        poseNet::DENSENET121_BASELINE_ATT_256x256
    );
    
    if(!net)
    {
        ROS_ERROR("failed to load poseNet model");
        return 0;
    }

    /*
    * create image converters
    */
    in_cvt = new imageConverter();
    
    

    if( !in_cvt )
    {
        ROS_ERROR("failed to create imageConverter objects");
        return 0;
    }
    
    /*
    * advertise publisher topics
    */
    ros::Publisher pose_publsh = private_nh.advertise<sensor_msgs::Image>("posenet_result", 2);
    pose_pub = &pose_publsh;


    /*
    * subscribe to image topic
    */
    ros::Subscriber img_sub = nh.subscribe(
        "/csi_cam_0/image_raw",
        5,
        img_callback
    );

    /*
    * wait for messages
    */
    ROS_INFO("PoseNet Node initialized, waiting for messages");

    ros::spin();

    return 0;
}
