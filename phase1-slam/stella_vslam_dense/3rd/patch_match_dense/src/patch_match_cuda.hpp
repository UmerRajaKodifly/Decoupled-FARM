#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/core.hpp>

extern "C"
{

void ComputeBearingsEquirect(float* d_Bearingmap, const int width, const int height);
void ComputeRandomPlanemapDepthmap(float* d_Depthmap, float* d_Planemap,float* d_Bearingmap,
	int width, int height, float min_depth, float max_depth, curandState* d_StatesMap);
void ComputeRandomPlanemap(float* d_Planemap, float* d_Depthmap, float* d_Bearingmap, int width,
	int height, curandState* d_StatesMap);
void ScorePlaneDepth(const int patchhalf, float* d_Scoremap, float* d_Depthmap, float* d_Planemap,
	float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height);
void PatchMatchRedBlackPass(const int patchhalf, int nIterations, float* d_Scoremap,
	float* d_Depthmap, float* d_Planemap, float* d_Bearingmap, cudaTextureObject_t* d_texs,
	int width, int height, curandState* d_StatesMap);
void ComputeRandomSeed(curandState* d_StatesMap, const int width, const int height);
void RandomInitialization(const int patchhalf, float* d_Scoremap, float* d_Depthmap,
	float* d_Planemap, float* d_Bearingmap, cudaTextureObject_t* d_texs, int width, int height,
	float min_depth, float max_depth, curandState* d_StatesMap);
void InitTextureImages(const int width, const int height, cv::Mat &Img1, cv::Mat &Img2,
	cv::Mat &Img3, float** d_img1, float** d_img2, float** d_img3, cudaTextureObject_t texs[]);
void InitTextureBearingTable(const int width, const int height, float* d_Bearingmap,
	float* d_Bearingmap_pitch, cudaTextureObject_t* bearing_tex);
void printDevProp(cudaDeviceProp devProp);
void InitConstantMem(float* R12, float* t12, float* R13, float* t13, float* R1_inv, float* t1_inv);

}