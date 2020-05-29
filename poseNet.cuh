#ifndef __POSE_NET_PREPROCESSING_H__
#define __POSE_NET_PREPROCESSING_H__

#include "Tensor.h"

#include <jetson-utils/cudaUtility.h>



/*
 * Downsample to RGB or BGR, NCHW format
 */
cudaError_t cudaPreImageNetRGB(
    float4* input,
    size_t inputWidth,
    size_t inputHeight,
    float* output,
    size_t outputWidth,
    size_t outputHeight,
    cudaStream_t stream
);


cudaError_t cudaPreImageNetBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );

/*
 * Downsample and apply mean pixel subtraction, NCHW format
 */
cudaError_t cudaPreImageNetMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );
cudaError_t cudaPreImageNetMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, NCHW format
 */
cudaError_t cudaPreImageNetNormRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );
cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );

/*
 * Downsample and apply pixel normalization, mean pixel subtraction and standard deviation, NCHW format
 */
cudaError_t cudaPreImageNetNormMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream );
cudaError_t cudaPreImageNetNormMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream );


__global__ void gpuDrawCircle(
    float4* output,
    const int out_width,
    const int out_height,
    const int center_x,
    const int center_y,
    const int radius
);


__global__ void gpuDrawLine(
    float4* output,
    const int out_width,
    const int out_height,
    const float x0,
    const float y0,
    const float x1,
    const float y1
);

cudaError_t cudaDrawPose(
    float4* output,
    uint32_t out_width,
    uint32_t out_height,
    jetsoncam::Tensor<int> topology,
    jetsoncam::Tensor<int> object_counts,
    jetsoncam::Tensor<int> objects,
    jetsoncam::Tensor<float> normalized_peaks,
    cudaStream_t stream
);

#endif
