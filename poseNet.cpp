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
 
#include "poseNet.h"
#include "poseNet.cuh"

#include "Tensor.h"
#include "ParseObjects.hpp"
#include "plugins.hpp"

#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaOverlay.h>
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/cudaFont.h>

#include <jetson-utils/commandLine.h>
#include <jetson-utils/filesystem.h>
#include <jetson-utils/imageIO.h>

#include <vector>

#define OUTPUT_CMAP  0 // CMAP
#define OUTPUT_PAF   1 // PAF

// constructor
poseNet::poseNet() : tensorNet()
{
    mLastInputImg    = NULL;
    mLastInputWidth  = 0;
    mLastInputHeight = 0;

    mClassColors[0] = NULL;    // cpu ptr
    mClassColors[1] = NULL;  // gpu ptr

    mClassMap[0] = NULL;
    mClassMap[1] = NULL;

    mNetworkType = POSENET_CUSTOM;

    topology_supercategory = "person";
    topology_id = 1;
    topology_name = "person";
    topology_keypoints = {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "neck"
    };

    // original topology
    topology_skeleton = {
        { 16, 14 },
        { 14, 12 },
        { 17, 15 },
        { 15, 13 },
        { 12, 13 },
        { 6, 8 },
        { 7, 9 },
        { 8, 10 },
        { 9, 11 },
        { 2, 3 },
        { 1, 2 },
        { 1, 3 },
        { 2, 4 },
        { 3, 5 },
        { 4, 6 },
        { 5, 7 },
        { 18, 1 },
        { 18, 6 },
        { 18, 7 },
        { 18, 12 },
        { 18, 13 }
    };

    /*
    // modified topology
    topology_skeleton = {
        { 15, 13 },
        { 13, 11 },
        { 16, 14 },
        { 14, 12 },
        { 11, 12 },
        { 5, 7 },
        { 6, 8 },
        { 7, 9 },
        { 8, 10 },
        { 1, 2 },
        { 0, 1 },
        { 0, 2 },
        { 1, 3 },
        { 2, 4 },
        { 3, 5 },
        { 4, 6 },
        { 17, 0 },
        { 17, 5 },
        { 17, 6 },
        { 17, 11 },
        { 17, 12 }
    };
    */

    num_parts = static_cast<int>(topology_keypoints.size());
    num_links = static_cast<int>(topology_skeleton.size());
    
    topology = trt_pose::plugins::coco_category_to_topology(topology_skeleton);
    topology.printDims();
    NetworkOutputParser = trt_pose::ParseObjects(topology);
}


// destructor
poseNet::~poseNet()
{
    
}

// NetworkTypeFromStr
poseNet::NetworkType poseNet::NetworkTypeFromStr( const char* modelName )
{
    if( !modelName )
        return poseNet::POSENET_CUSTOM;

    poseNet::NetworkType type = poseNet::DENSENET121_BASELINE_ATT_256x256;

    // ONNX models
    if( strcasecmp(modelName, "densenet121_baseline_att_256x256_B_epoch_160") == 0 || strcasecmp(modelName, "densenet121_baseline_att") == 0 )
        type = poseNet::DENSENET121_BASELINE_ATT_256x256;
    else if( strcasecmp(modelName, "resnet18_baseline_att_224x224_A_epoch_249") == 0 || strcasecmp(modelName, "resnet18_baseline_att") == 0 )
        type = poseNet::RESNET18_BASELINE_ATT_224x224;
    else
        type = poseNet::POSENET_CUSTOM;

    return type;
}


// NetworkTypeToStr
const char* poseNet::NetworkTypeToStr( poseNet::NetworkType type )
{
    switch(type)
    {
        // ONNX models
        case DENSENET121_BASELINE_ATT_256x256:    return "densenet121_baseline_att_256x256_B_epoch_160";
        case RESNET18_BASELINE_ATT_224x224:    return "resnet18_baseline_att_224x224_A_epoch_249";
        default:                            return "custom poseNet";
    }
}


// Create
poseNet* poseNet::Create(
    NetworkType networkType,
    uint32_t maxBatchSize,
    precisionType precision,
    deviceType device,
    bool allowGPUFallback
){
    poseNet* net = NULL;

    // ONNX models
    if( networkType == DENSENET121_BASELINE_ATT_256x256 ) {
        net = Create(
            NULL,
            "networks/densenet121_baseline_att_256x256_B_epoch_160.onnx",
            "0",
            "1227",
            "1229",
            maxBatchSize,
            precision,
            device,
            allowGPUFallback
        );
    }
    else if( networkType == RESNET18_BASELINE_ATT_224x224 ) {
        net = Create(
            NULL,
            "networks/resnet18_baseline_att_224x224_A_epoch_249.onnx",
            "0",
            "262",
            "264",
            maxBatchSize,
            precision,
            device,
            allowGPUFallback
        );
    }
    else {
        return NULL;
    }

    if( net != NULL )
        net->mNetworkType = networkType;

    return net;
}


// Create
poseNet* poseNet::Create(
    const char* prototxt,
    const char* model,
    const char* input_blob,
    const char* cmap_blob,
    const char* paf_blob,
    uint32_t maxBatchSize,
    precisionType precision,
    deviceType device,
    bool allowGPUFallback
){
    // create segmentation model
    poseNet* net = new poseNet();
    
    if( !net )
        return NULL;

    printf("\n");
    printf("poseNet -- loading segmentation network model from:\n");
    printf("       -- prototxt:   %s\n", prototxt);
    printf("       -- model:      %s\n", model);
    printf("       -- input_blob  '%s'\n", input_blob);
    printf("       -- cmap_blob '%s'\n", cmap_blob);
    printf("       -- paf_blob '%s'\n", paf_blob);
    printf("       -- batch_size  %u\n\n", maxBatchSize);
    
    //net->EnableProfiler();    
    //net->EnableDebug();
    //net->DisableFP16();        // debug;

    // load network
    std::vector<std::string> output_blobs;
    output_blobs.push_back(cmap_blob);
    output_blobs.push_back(paf_blob);
    
    if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, maxBatchSize,
                      precision, device, allowGPUFallback) )
    {
        printf("poseNet -- failed to initialize.\n");
        return NULL;
    }

    return net;
}


// Pre Process and Classify the image
bool poseNet::Process( float* rgba, uint32_t width, uint32_t height )
{
    if( !rgba || width == 0 || height == 0 )
    {
        printf("poseNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
        return false;
    }

    PROFILER_BEGIN(PROFILER_PREPROCESS);

    if( IsModelType(MODEL_ONNX) )
    {
        // downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
        if( CUDA_FAILED(cudaPreImageNetNormMeanRGB(
            (float4*)rgba,
            width,
            height,
            mInputCUDA,
            mWidth,
            mHeight, 
            make_float2(0.0f, 1.0f), // range
            make_float3(0.485f, 0.456f, 0.406f), // mean
            make_float3(0.229f, 0.224f, 0.225f),  // stdDev
            GetStream())) )
        {
            printf(LOG_TRT "poseNet::Process() -- cudaPreImageNetNormMeanRGB() failed\n");
            return false;
        }
    }
    else
    {
        // downsample and convert to band-sequential BGR
        if( CUDA_FAILED(cudaPreImageNetBGR(
            (float4*)rgba,
            width,
            height,
            mInputCUDA,
            mWidth,
            mHeight,
            GetStream())) )
        {
            printf("poseNet::Process() -- cudaPreImageNetBGR() failed\n");
            return false;
        }
    }

    PROFILER_END(PROFILER_PREPROCESS);
    PROFILER_BEGIN(PROFILER_NETWORK);
    
    // process with TensorRT
    void* inferenceBuffers[] = { mInputCUDA, mOutputs[OUTPUT_CMAP].CUDA, mOutputs[OUTPUT_PAF].CUDA };
    
    // execute the neural network with your image input
    if( !mContext->execute(1, inferenceBuffers) )
    {
        printf(LOG_TRT "poseNet::Process() -- failed to execute TensorRT context\n");
        return false;
    }

    PROFILER_END(PROFILER_NETWORK);
    PROFILER_BEGIN(PROFILER_POSTPROCESS);

    printf("width: %u, height: %u, mWidth: %u, mHeight: %u\n", width, height, mWidth, mHeight);
    // process model Output
    if( !processOutput(rgba, width, height) )
        return false;

    PROFILER_END(PROFILER_POSTPROCESS);

    // cache pointer to last image processed
    mLastInputImg = rgba;
    mLastInputWidth = width;
    mLastInputHeight = height;

    return true;
}


// processOutput
bool poseNet::processOutput(
    float* output,
    uint32_t width,
    uint32_t height
)
{
    size_t outputLen = mOutputs.size();

    for (size_t i = 0; i < outputLen; i++) {
        const char* output_name = mOutputs[i].name.c_str();
        printf(LOG_TRT "poseNet::processOutput() : %s \n", output_name);
    }

    // retrieve scores
    float* cmap = mOutputs[OUTPUT_CMAP].CPU;
    float* paf = mOutputs[OUTPUT_PAF].CPU;

    const int c_w = DIMS_W(mOutputs[OUTPUT_CMAP].dims);
    const int c_h = DIMS_H(mOutputs[OUTPUT_CMAP].dims);
    const int c_c = DIMS_C(mOutputs[OUTPUT_CMAP].dims);
    
    const int p_w = DIMS_W(mOutputs[OUTPUT_PAF].dims);
    const int p_h = DIMS_H(mOutputs[OUTPUT_PAF].dims);
    const int p_c = DIMS_C(mOutputs[OUTPUT_PAF].dims);
    
    // data/image:: torch.Size([1, 3, 224, 224])

    jetsoncam::Tensor<float> cmap_tensor = jetsoncam::Tensor<float>(
        "cmap_tensor",
        {1, c_c, c_h, c_w},
        mOutputs[OUTPUT_CMAP].CPU,
        mOutputs[OUTPUT_CMAP].CUDA
    );
    cmap_tensor.printDims();
    
    jetsoncam::Tensor<float> paf_tensor = jetsoncam::Tensor<float>(
        "paf_tensor",
        {1, p_c, p_h, p_w},
        mOutputs[OUTPUT_PAF].CPU,
        mOutputs[OUTPUT_PAF].CUDA
    );
    paf_tensor.printDims();
    
    // cmap:: torch.Size([1, 18, 56, 56]) [Correct]
    // Tensor cmap_tensor 4 dimensions { 1  18  56  56 } [Correct]
    // paf:: torch.Size([1, 42, 56, 56]) [Correct]
    // Tensor paf_tensor 4 dimensions { 1  42  56  56 } [Correct]
    
    jetsoncam::ParseResult networkResults = NetworkOutputParser.Parse(cmap_tensor, paf_tensor);

    jetsoncam::Tensor<int> object_counts = networkResults.object_counts;
    object_counts.printDims();
    jetsoncam::Tensor<int> objects = networkResults.objects;
    objects.printDims();
    jetsoncam::Tensor<float> normalized_peaks = networkResults.normalized_peaks;
    normalized_peaks.printDims();
    
    // counts:: torch.Size([1]) [Correct]
    // Tensor object_counts 1 dimensions { 1 } [Correct]
    // objects:: torch.Size([1, 100, 18]) [Correct]
    // Tensor objects 3 dimensions { 1  100  18 } [Correct]
    // peaks:: torch.Size([1, 18, 100, 2]) [Correct]
    // Tensor refined_peaks 4 dimensions { 1  18  100  2 } [Correct]
    
    printf(LOG_TRT "poseNet::processOutput() Computed\n");
    printf(LOG_TRT "    ----- object_counts\n");
    printf(LOG_TRT "    ----- objects\n");
    printf(LOG_TRT "    ----- normalized_peaks\n");

    return overlayPosePoints(
        output,
        width,
        height,
        topology,
        object_counts,
        objects,
        normalized_peaks
    );
}

#define OVERLAY_CUDA 

// overlayLinear
bool poseNet::overlayPosePoints(
    float* input,
    uint32_t width,
    uint32_t height,
    jetsoncam::Tensor<int> topology,
    jetsoncam::Tensor<int> object_counts,
    jetsoncam::Tensor<int> objects,
    jetsoncam::Tensor<float> normalized_peaks
)
{
    PROFILER_BEGIN(PROFILER_VISUALIZE);

#ifdef OVERLAY_CUDA
    // uint8_t* scores;
    // generate overlay on the GPU
    if( CUDA_FAILED(cudaDrawPose(
        (float4*)input,
        width,
        height,
        topology,
        object_counts,
        objects,
        normalized_peaks,
        GetStream())) )
    {
        printf(LOG_TRT "poseNet -- failed to process %ux%u overlay/mask with CUDA\n", width, height);
        return false;
    }
#endif
    PROFILER_END(PROFILER_VISUALIZE);
    
    printf(LOG_TRT "poseNet -- completed Drawing Pose\n");
    return true;
}

//*********************** PYTHON NOTEBOOK EXECUTION STEPS ***********************

//     draw_objects(image, counts, objects, peaks)
//     image_w.value = bgr8_to_jpeg(image[:, ::-1, :])