#include "include/CImg.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "include/processes.h"

using namespace std;
using namespace cimg_library;



static tImageInfo processInfoGPU;

//getting a copy of image information on the GPU SIDE
void imageInitGPU(tImageInfo& information)
{
	processInfoGPU = information;
}

void setProcessGPU(eProcess process)
{
	processInfoGPU.processType = process;
}


__global__ void convolveKernel(unsigned char* d_image, unsigned char* d_output, float* d_kernel, int width, int height, int kernelSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int kCenter = kernelSize / 2;
	

	//on each pixel we are still in bounds...
	if (x < width && y < height)
	{
		//one itteration per RGB channel
		for (int c = 0; c < 3; c++) {

			float value = 0.0f;
			
			//applying convoluition at each pixel ---> each pixel has an associated thread above
			for (int ky = 0; ky < kernelSize; ky++) {
				for (int kx = 0; kx < kernelSize; kx++) {
					
					//getting relative neighbouring pixel coordinates in the kernal 
					int px = x + (kx - kCenter);
					int py = y + (ky - kCenter);

					//accumulating result of convulution calculation
					if (px >= 0 && px < width && py >= 0 && py < height) {
						value += d_kernel[ky * kernelSize + kx] * d_image[(py * width + px) * 3 + c];
					}
				}
			}
			//clamping value from [0, 255]
			d_output[(y * width + x) * 3 + c] = min(max(int(value), 0), 255);
		}
	}
}



CImg<unsigned char> applyKernelCUDA(const vector<vector<float>>& kernel, const CImg<unsigned char>& image)
{

	//init everything needed to make new image
	int width = image.width();
	int height = image.height();
	int kernelSize = kernel.size();
	CImg<unsigned char> newImage(width, height, 1, 3, 0);

	//flattening the kernel to store it in GPU memory easier ---> MUST BE HANDLED PROPERLY IN CONVOLUTION CODE
	vector<float> flatKernel(kernelSize * kernelSize);
	for (int y = 0; y < kernelSize; ++y) {
		for (int x = 0; x < kernelSize; ++x) {

			//going from 2d to 1d ----> y * (rowSize) + x_offset
			flatKernel[y * kernelSize + x] = kernel[y][x];
		}
	}

	//preparing pointers for GPU meomory
	unsigned char* d_image;
	unsigned char* d_output;
	float* d_kernel;

	//getting value to represent size of these values in memory
	size_t imageSize = width * height * 3 * sizeof(unsigned char);
	size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

	//allocating space on gpu
	cudaMalloc(&d_image, imageSize);
	cudaMalloc(&d_output, imageSize);
	cudaMalloc(&d_kernel, kernelSizeBytes);

	//copying to the gpu
	cudaMemcpy(d_image, image.data(), imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, flatKernel.data(), kernelSizeBytes, cudaMemcpyHostToDevice);

	//defining dim and block space and LAUNCHING the kernel
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	convolveKernel <<<gridSize, blockSize>>> (d_image, d_output, d_kernel, width, height, kernelSize);

	//synching data
	cudaDeviceSynchronize();

	//copy the date back to the new image
	cudaMemcpy(newImage.data(), d_output, imageSize, cudaMemcpyDeviceToHost);

	//free all memory allocated on GPU earlier
	cudaFree(d_image);
	cudaFree(d_output);
	cudaFree(d_kernel);

	return newImage;
}












