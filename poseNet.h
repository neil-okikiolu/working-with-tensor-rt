/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef __POSE_NET_H__
#define __POSE_NET_H__

#include "Tensor.h"
#include "ParseObjects.hpp"

#include <jetson-inference/tensorNet.h>
#include <vector>


/**
 * Name of default input blob for segmentation model.
 * @ingroup poseNet
 */
#define POSENET_DEFAULT_INPUT   "Input"

/**
 * Name of default output blob for segmentation model.
 * @ingroup poseNet
 */
#define RESNET_DEFAULT_CMAP_OUTPUT  "262"
#define RESNET_DEFAULT_PAF_OUTPUT  "264"

#define DENSENET_DEFAULT_CMAP_OUTPUT  "262"
#define DENSENET_DEFAULT_PAF_OUTPUT  "264"

/**
 * Command-line options able to be passed to poseNet::Create()
 * @ingroup poseNet
 */
#define POSENET_USAGE_STRING  "poseNet arguments: \n"                             \
          "  --network NETWORK    pre-trained model to load, one of the following:\n"     \
          "                           * fcn-resnet18-cityscapes-512x256\n"            \
          "                           * fcn-resnet18-cityscapes-1024x512\n"            \
          "  --model MODEL        path to custom model to load (caffemodel, uff, or onnx)\n"             \
          "  --input_blob INPUT   name of the input layer (default: '" POSENET_DEFAULT_INPUT "')\n"         \
          "  --output_blob OUTPUT name of the output layer (default: '" RESNET_DEFAULT_CMAP_OUTPUT "')\n"         \
          "  --batch_size BATCH   maximum batch size (default is 1)\n"                                \
          "  --profile            enable layer profiling in TensorRT\n"

/**
 * Image segmentation with FCN-Alexnet or custom models, using TensorRT.
 * @ingroup poseNet
 */
class poseNet : public tensorNet
{
public:
    /**
     * Enumeration of pretrained/built-in network models.
     */
    enum NetworkType
    {
        DENSENET121_BASELINE_ATT_256x256,
        RESNET18_BASELINE_ATT_224x224,
        
        /* add new models here */
        POSENET_CUSTOM
    };

    /**
     * Parse a string from one of the built-in pretrained models.
     * Valid names are "cityscapes-hd", "cityscapes-sd", "pascal-voc", ect.
     * @returns one of the poseNet::NetworkType enums, or poseNet::CUSTOM on invalid string.
     */
    static NetworkType NetworkTypeFromStr( const char* model_name );

    /**
     * Convert a NetworkType enum to a human-readable string.
     * @returns stringized version of the provided NetworkType enum.
     */
    static const char* NetworkTypeToStr( NetworkType networkType );


    /**
     * Load a new network instance
     */
    static poseNet* Create(
        NetworkType networkType=RESNET18_BASELINE_ATT_224x224,
        uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE,
        precisionType precision=TYPE_FASTEST,
        deviceType device=DEVICE_GPU,
        bool allowGPUFallback=true
    );
    
    /**
     * Load a new network instance
     * @param prototxt_path File path to the deployable network prototxt
     * @param model_path File path to the caffemodel
     * @param class_labels File path to list of class name labels
     * @param class_colors File path to list of class colors
     * @param input Name of the input layer blob. @see POSENET_DEFAULT_INPUT
     * @param cmap_blob Name of the cmap output layer blob. @see RESNET_DEFAULT_CMAP_OUTPUT
     * @param paf_blob Name of the paf output layer blob. @see RESNET_DEFAULT_PAF_OUTPUT
     * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
     */
    static poseNet* Create(
        const char* prototxt_path,
        const char* model_path, 
        const char* input = POSENET_DEFAULT_INPUT, 
        const char* cmap_blob = RESNET_DEFAULT_CMAP_OUTPUT,
        const char* paf_blob = RESNET_DEFAULT_PAF_OUTPUT,
        uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
        precisionType precision=TYPE_FASTEST, 
        deviceType device=DEVICE_GPU,
        bool allowGPUFallback=true
    );
    
    /**
     * Usage string for command line arguments to Create()
     */
    static inline const char* Usage()         { return POSENET_USAGE_STRING; }

    /**
     * Destroy
     */
    virtual ~poseNet();
    
    /**
      * Perform the initial inferencing processing portion of the segmentation.
     * The results can then be visualized using the Overlay() and Mask() functions.
     * @param input float4 input image in CUDA device memory, RGBA colorspace with values 0-255.
     * @param width width of the input image in pixels.
     * @param height height of the input image in pixels.
     */
    bool Process( float* input, uint32_t width, uint32_t height );

    /**
     * Retrieve the number of columns in the classification grid.
     * This indicates the resolution of the raw segmentation output.
     */
    inline uint32_t GetGridWidth() const                        { return DIMS_W(mOutputs[0].dims); }

    /**
     * Retrieve the number of rows in the classification grid.
     * This indicates the resolution of the raw segmentation output.
     */
    inline uint32_t GetGridHeight() const                        { return DIMS_H(mOutputs[0].dims); }

    /**
     * Retrieve the network type (alexnet or googlenet)
     */
    inline NetworkType GetNetworkType() const                    { return mNetworkType; }

    /**
      * Retrieve a string describing the network name.
     */
    inline const char* GetNetworkName() const                    { return NetworkTypeToStr(mNetworkType); }

protected:
    poseNet();
    
    bool processOutput(float* output, uint32_t width, uint32_t height);

    bool overlayPosePoints(
        float* input,
        uint32_t width,
        uint32_t height,
        jetsoncam::Tensor<int> topology,
        jetsoncam::Tensor<int> object_counts,
        jetsoncam::Tensor<int> objects,
        jetsoncam::Tensor<float> normalized_peaks
    );

    float*   mClassColors[2];    /**< array of overlay colors in shared CPU/GPU memory */
    uint8_t* mClassMap[2];        /**< runtime buffer for the argmax-classified class index of each tile */
    
    float*   mLastInputImg;        /**< last input image to be processed, stored for overlay */
    uint32_t mLastInputWidth;    /**< width in pixels of last input image to be processed */
    uint32_t mLastInputHeight;    /**< height in pixels of last input image to be processed */

    NetworkType mNetworkType;    /**< Pretrained built-in model type enumeration */

    const char* topology_supercategory;
    uint32_t topology_id;
    const char* topology_name;
    std::vector<const char*> topology_keypoints;
    std::vector<std::vector<int>> topology_skeleton;
    int num_parts;
    int num_links;
    jetsoncam::Tensor<int> topology;
    trt_pose::ParseObjects NetworkOutputParser;
};


#endif

