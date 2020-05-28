/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "poseNet.cuh"

#include "Tensor.h"

#include <jetson-utils/cudaUtility.h>
#include <cstdlib>
#include <cmath>



// gpuDrawCircle
__global__ void gpuDrawCircle(
    float4* output,
    const int out_width,
    const int out_height,
    const int center_x,
    const int center_y,
    const int radius
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= out_width || y >= out_height )
        return;
        
    // if x,y is in the circle draw it
    if ((x - center_x)*(x - center_x) + (y - center_y)*(y - center_y) < (radius * radius)) {
        const float4 color = make_float4(0.0f, 0.0f, 255.0f, 255.0f);
        output[y * out_width + x] = color;
    }
    
}


// gpuDrawLine
__global__ void gpuDrawLine(
    float4* output,
    const int out_width,
    const int out_height,
    const float x0,
    const float y0,
    const float x1,
    const float y1
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x >= out_width || y >= out_height )
        return;
      
    float AB = std::sqrt((x1-x0) * (x1-x0) + (y1-y0) * (y1-y0));
    float AP = std::sqrt((x-x0) * (x-x0) + (y-y0) * (y-y0));
    float PB = std::sqrt((x1-x) * (x1-x) + (y1-y) * (y1-y));

    // adjust threshold to make the line thicker
    const float threshold = 0.1f;
    if (std::fabs(AB - (AP + PB)) <= threshold) {
        const float4 color = make_float4(0.0f, 0.0f, 255.0f, 255.0f);
        output[y * out_width + x] = color;
    }
    
}

// cudaPreImageNet
cudaError_t cudaDrawPose(
    float4* output,
    uint32_t out_width,
    uint32_t out_height,
    jetsoncam::Tensor<int> topology,
    jetsoncam::Tensor<int> object_counts,
    jetsoncam::Tensor<int> objects,
    jetsoncam::Tensor<float> normalized_peaks,
    cudaStream_t stream
)
{
    if( !output )
        return cudaErrorInvalidDevicePointer;

    if( out_width == 0 || out_height == 0 )
        return cudaErrorInvalidValue;
        
    int K = topology.size(0);
    int count = object_counts.retrieve({0});
    //printf("count: %d\n", count);
    //printf("K: %d\n", K);
    //printf("output image width %u, height %u\n", out_width, out_height);
    
    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(out_height,blockDim.x), iDivUp(out_height,blockDim.y));
    
    for(int i = 0; i < count; i++)
    {
        int C = objects.size(2);
        for (int j = 0; j < C; j++) {
            int k = objects.retrieve({0,i,j});
            
            if (k >= 0) {
                float x = normalized_peaks.retrieve({0,j,k,1}) * float(out_width);
                float y = normalized_peaks.retrieve({0,j,k,0}) * float(out_height);
                // DRAW x,y to a circle with color
                gpuDrawCircle<<<gridDim, blockDim, 0, stream>>>(
                    output,
                    out_width,
                    out_height,
                    (int) x,
                    (int) y,
                    5
                );
            }
        }
        
        for (int k = 0; k < K; k++) {
            int c_a = topology.retrieve({k,2});
            int c_b = topology.retrieve({k,3});
            
            int obj_c_a = objects.retrieve({0,i,c_a});
            int obj_c_b = objects.retrieve({0,i,c_b});
            
            if (obj_c_a >= 0 && obj_c_b >= 0) {
                float x0 = normalized_peaks.retrieve({0,c_a,obj_c_a,1}) * float(out_width);
                float y0 = normalized_peaks.retrieve({0,c_a,obj_c_a,0}) * float(out_height);
                float x1 = normalized_peaks.retrieve({0,c_b,obj_c_b,1}) * float(out_width);
                float y1 = normalized_peaks.retrieve({0,c_b,obj_c_b,0}) * float(out_height);
                // printf("gpuDrawLine-> obj_c_a: %d, obj_c_b: %d, x0: %f, y0: %f, x1: %f, y1: %f\n",obj_c_a, obj_c_b, x0, y0, x1, y1);
                // DRAW line from x0,y0 to x1,y1
                gpuDrawLine<<<gridDim, blockDim, 0, stream>>>(
                    output,
                    out_width,
                    out_height,
                    x0,
                    y0,
                    x1,
                    y1
                );
            }
        }
    }

    return CUDA(cudaGetLastError());
}




// gpuPreImageNetRGB
__global__ void gpuPreImageNetRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = bgr.x;
	output[n * 1 + m] = bgr.y;
	output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetRGB
cudaError_t cudaPreImageNetRGB( float4* input, size_t inputWidth, size_t inputHeight,
				            float* output, size_t outputWidth, size_t outputHeight,
					       cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetBGR
__global__ void gpuPreImageNetBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + m] = bgr.x;
	output[n * 1 + m] = bgr.y;
	output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetBGR
cudaError_t cudaPreImageNetBGR( float4* input, size_t inputWidth, size_t inputHeight,
				            float* output, size_t outputWidth, size_t outputHeight,
					       cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetMeanRGB
__global__ void gpuPreImageNetMeanRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.x - mean_value.x, px.y - mean_value.y, px.z - mean_value.z);
	
	output[n * 0 + m] = bgr.x;
	output[n * 1 + m] = bgr.y;
	output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetMeanRGB
cudaError_t cudaPreImageNetMeanRGB( float4* input, size_t inputWidth, size_t inputHeight,
				                float* output, size_t outputWidth, size_t outputHeight, 
						      const float3& mean_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetMeanRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetMeanBGR
__global__ void gpuPreImageNetMeanBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z - mean_value.x, px.y - mean_value.y, px.x - mean_value.z);
	
	output[n * 0 + m] = bgr.x;
	output[n * 1 + m] = bgr.y;
	output[n * 2 + m] = bgr.z;
}


// cudaPreImageNetMeanBGR
cudaError_t cudaPreImageNetMeanBGR( float4* input, size_t inputWidth, size_t inputHeight,
				                float* output, size_t outputWidth, size_t outputHeight, 
						      const float3& mean_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetMeanBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormRGB
__global__ void gpuPreImageNetNormRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = bgr.x * multiplier + min_value;
	output[n * 1 + m] = bgr.y * multiplier + min_value;
	output[n * 2 + m] = bgr.z * multiplier + min_value;
}


// cudaPreImageNetNormRGB
cudaError_t cudaPreImageNetNormRGB( float4* input, size_t inputWidth, size_t inputHeight,
							 float* output, size_t outputWidth, size_t outputHeight,
							 const float2& range, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	//printf("cudaPreImageNetNorm([%f, %f])  multiplier=%f\n", range.x, range.y, multiplier);
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetNormRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormBGR
__global__ void gpuPreImageNetNormBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + m] = bgr.x * multiplier + min_value;
	output[n * 1 + m] = bgr.y * multiplier + min_value;
	output[n * 2 + m] = bgr.z * multiplier + min_value;
}


// cudaPreImageNetNorm
cudaError_t cudaPreImageNetNormBGR( float4* input, size_t inputWidth, size_t inputHeight,
								 float* output, size_t outputWidth, size_t outputHeight,
								 const float2& range, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	//printf("cudaPreImageNetNorm([%f, %f])  multiplier=%f\n", range.x, range.y, multiplier);
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetNormBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);

	return CUDA(cudaGetLastError());
}



// gpuPreImageNetNormMeanRGB
__global__ void gpuPreImageNetNormMeanRGB( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.x * multiplier + min_value, px.y * multiplier + min_value, px.z * multiplier + min_value);
	
	output[n * 0 + m] = (bgr.x - mean.x) / stdDev.x;
	output[n * 1 + m] = (bgr.y - mean.y) / stdDev.y;
	output[n * 2 + m] = (bgr.z - mean.z) / stdDev.z;
}


// cudaPreImageNetNormMeanRGB
cudaError_t cudaPreImageNetNormMeanRGB( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetNormMeanRGB<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x, mean, stdDev);

	return CUDA(cudaGetLastError());
}


// gpuPreImageNetNormMeanBGR
__global__ void gpuPreImageNetNormMeanBGR( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z * multiplier + min_value, px.y * multiplier + min_value, px.x * multiplier + min_value);
	
	output[n * 0 + m] = (bgr.x - mean.x) / stdDev.x;
	output[n * 1 + m] = (bgr.y - mean.y) / stdDev.y;
	output[n * 2 + m] = (bgr.z - mean.z) / stdDev.z;
}


// cudaPreImageNetNormMeanBGR
cudaError_t cudaPreImageNetNormMeanBGR( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, const float3& mean, const float3& stdDev, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetNormMeanBGR<<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x, mean, stdDev);

	return CUDA(cudaGetLastError());
}



