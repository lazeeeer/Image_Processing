#ifndef PROCESSES_H
#define PROCESSES_H

#include <iostream>
#include "CImg.h"
#include <vector>
#include <cuda_runtime.h>

using namespace std;
using namespace cimg_library;


//enums of different process types
typedef enum{

    eNone = 0,
    eBoxBlur,
    eSharpen

}eProcess;


// struct of image information
typedef struct {

    std::string img_path;
    eProcess processType;
    int kernel_size;

    // CImg object to hold image info on curr working process
    CImg<unsigned char> workingImg;

}tImageInfo;


/* ==== ALL CPU RELATED FUNCTIONS FROM process.cpp ====  */

// Misc fucntions for inti, debug, etc...
void imageInit(std::string& filePath, eProcess effect);
void dispImage();

//getter and setter functions for process struct
eProcess getProcess();
void setProcess(eProcess process);
std::string getDir();
std::string getProcessName();
//unused?????
void setKernel(int n);
int getKernel();

//box blur alg function calls
unsigned char boxBlurCalc(int y,int x, int channel);
void boxBlur(CImg<unsigned char>(*applyKernel)(const vector<vector<float>>& kernel, const CImg<unsigned char>& image));

//sharpen function calls
void sharpen(CImg<unsigned char>(*applyKernel)(const vector<vector<float>>& kernel, const CImg<unsigned char>& image));

//CPU convolution calls
CImg<unsigned char> applyKernelCPU(const vector<vector<float>>& kernel, const CImg<unsigned char>& image);
unsigned char convolveCalc(int x, int y, int channel, const CImg<unsigned char>& image, const vector<vector<float>>& kernel);



/* ==== ALL CPU RELATED FUNCTIONS FROM process.cpp ====  */

//misc functions for init and debug
void imageInitGPU(tImageInfo& information);

//getter and setter functions for struct information
void setProcessGPU(eProcess process);

//GPU convolution calls
CImg<unsigned char> applyKernelCUDA(const vector<vector<float>>& kernel, const CImg<unsigned char>& image);
__global__ void convolveKernel(unsigned char* d_image, unsigned char* d_output, float* d_kernel, int width, int height, int kernelSize);

#endif // FUNCTIONS_H
