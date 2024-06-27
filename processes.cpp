#include "include/CImg.h"
#include <iostream>
#include <vector>

#include "include/processes.h"

using namespace std;
using namespace cimg_library;

// static declaration for struct
static tImageInfo processInfo;


// init function for function Information
void imageInit(string& filePath, eProcess effect)
{
    
    processInfo.img_path = filePath;
    processInfo.processType = effect;
    processInfo.kernel_size = 3;

    //assigning constructors image variabile to the image path inputted
    processInfo.workingImg.assign( filePath.c_str() );

    //passing a copy of this struct to the GPU code so it can do the same processing...
    imageInitGPU(processInfo);

    //cout << processInfo.img_path << endl;
    //cout << processInfo.processType << endl;
}


// simple function to disp image
void dispImage()
{
    //calling CImg display using curr processes image object
    CImgDisplay display(processInfo.workingImg, "User Image");

    //waiting for user to close display
    while(!display.is_closed()){
        display.wait();
    }
}


//General function for image convolution
CImg<unsigned char> applyKernelCPU(const vector<vector<float>>& kernel, const CImg<unsigned char>& image)
{

    //getting values of width and height
    int width = processInfo.workingImg.width();
    int height = processInfo.workingImg.height();
    
    CImg<unsigned char> newImage(width, height, 1, 3, 0);

    for(int y = 1; y < height-1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
			newImage(x, y, 0, 0) = convolveCalc(x, y, 0, image, kernel);
			newImage(x, y, 0, 1) = convolveCalc(x, y, 1, image, kernel);
			newImage(x, y, 0, 2) = convolveCalc(x, y, 2, image, kernel);
        }
    }

    //newImage.save("C:/Vs2022/ImageProcessing/ImageProcessing/ImageProcessing/images/newImage.jpg");

    return newImage;
}


//helper function for making convolution calculations
unsigned char convolveCalc(int x, int y, int channel, const CImg<unsigned char>& image, const vector<vector<float>>& kernel)
{
    int kernelSize = kernel.size();
    int kCenter = kernelSize / 2;
    double value = 0;

    for (int ky = 0; ky < kernelSize; ky++)
    {
        for (int kx = 0; kx < kernelSize; kx++)
        {
            int px = x + (kx - kCenter);
            int py = y + (ky - kCenter);
            value += kernel[ky][kx] * image(px, py, 0, channel);
        }
    }

    return static_cast<unsigned char>(std::min(std::max(int(value), 0), 255));
}


void sharpen( CImg<unsigned char> (*applyKernel)(const vector<vector<float>>& kernel, const CImg<unsigned char>& image) )
{
    //kernel for getting detail image
	const vector<vector<float>> lapKernel = {
		{-1, -1, -1},
		{-1,  8, -1},
		{-1, -1, -1}
	};

    CImg<unsigned char> detailImg = applyKernel(lapKernel, processInfo.workingImg);
    detailImg.save("C:/Vs2022/ImageProcessing/ImageProcessing/ImageProcessing/images/detailImage.jpg");
    
    int height = processInfo.workingImg.height();
    int width = processInfo.workingImg.width();

    //defining new result image
    CImg<unsigned char> sharpenedImg(width, height, 1, 3, 0);


    //summing the two detail image and original image for sharppning...
    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++)
        {
            //need to do calc for each RGB channel of the image
            for (int channel = 0; channel < 3; channel++)
            {
                int originalValue   = processInfo.workingImg(x, y, 0, channel);
                int detailValue     = detailImg(x, y, 0, channel);
                int sharpenedValue  = originalValue + detailValue;

                // Clamp the result to the range [0, 255]
                sharpenedImg(x, y, 0, channel) = static_cast<unsigned char>(std::min(std::max(sharpenedValue, 0), 255));
            }
        }
    }

    //saving final image
    sharpenedImg.save("C:/Vs2022/ImageProcessing/ImageProcessing/ImageProcessing/images/sharpenedImage.jpg");
}



void boxBlur(CImg<unsigned char>(*applyKernel)(const vector<vector<float>>& kernel, const CImg<unsigned char>& image))
{

    //kernel for getting detail image
	const vector<vector<float>> boxBlurKernel = {
        {0.111, 0.111, 0.111},
        {0.111, 0.111, 0.111},
        {0.111, 0.111, 0.111}
	};

    CImg<unsigned char> blurredImg = applyKernel(boxBlurKernel, processInfo.workingImg);
    blurredImg.save("C:/Vs2022/ImageProcessing/ImageProcessing/ImageProcessing/images/blurredImage.jpg");
}



//getting information struct
tImageInfo* getImageInfo()
{
    return &processInfo;
}


// process getters and setters
eProcess getProcess()
{
    return processInfo.processType;
}


void setProcess(eProcess process)
{
    processInfo.processType = process;
}


// file path and name getters
std::string getDir()
{
    return processInfo.img_path;
}

int getKernel()
{
    return processInfo.kernel_size;
}


void setKernel(int n)
{
    processInfo.kernel_size = n;
}


std::string getProcessName()
{
    if (processInfo.processType == eNone){
        return "None";
    }
    else if (processInfo.processType == eBoxBlur){
        return "Box Blur";
    }
    else if (processInfo.processType == eSharpen){
        return "Sharpening";
    }

    return "ERROR";
}
