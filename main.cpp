#include "include/CImg.h"
#include <iostream>
#include <string>
#include <cstdlib>

#include "include/processes.h"

using namespace std;
using namespace cimg_library; 

// funciton call to print program menu screen
void printMenu()
{
    std::cout << "\033[H\033[J"; // Clear screen
    cout << "\n=======================================================================================================" << endl;
    cout << R"(

                        .____                                                            
                        |    |   _____  ________ ___________                             
                        |    |   \__  \ \___   // __ \_  __ \                            
                        |    |___ / __ \_/    /\  ___/|  | \/                            
                        |_______ (____  /_____ \\___  >__|                               
                                \/    \/      \/    \/                                   
                        .___                                                             
                        |   | _____ _____     ____   ____                                
                        |   |/     \\__  \   / ___\_/ __ \                               
                        |   |  Y Y  \/ __ \_/ /_/  >  ___/                               
                        |___|__|_|  (____  /\___  / \___  >                              
                                \/     \//_____/      \/                               
                        __________                                                       
                        \______   \_______  ____   ____  ____   ______ _________________ 
                        |     ___/\_  __ \/  _ \_/ ___\/ __ \ /  ___//  ___/  _ \_  __ \
                        |    |     |  | \(  <_> )  \__\  ___/ \___ \ \___ (  <_> )  | \/
                        |____|     |__|   \____/ \___  >___  >____  >____  >____/|__|   
                                                    \/    \/     \/     \/             
    )";

    cout << "" << endl;
    printf("Current Image   : %s \n", getDir().c_str());
    cout << "\n";

    std::cout << "Choose an option:\n\n";

    std::cout << "1. Check Selected Image\n";
    std::cout << "2. Select Image Blur\n";
    std::cout << "3. Select Image Sharpener\n";
    std::cout << "4. Select Kernel Size\n";
    std::cout << "5. Begin Processing Image\n";
    std::cout << "6. Quit";

    cout << "\n\n=======================================================================================================\n" << endl;
    std::cout << "Enter your choice below :\n\n";
    cout << "=======================================================================================================\n\n\n" << endl;
}


void moveToInputArea()
{
    std::cout << "\033[42;1H"; // Move cursor to the 43rd row, 1st column
}

void clearInputArea()
{
    std::cout << "\033[J"; // Clear from cursor to end of screen
}


//main program entry point
int main(int argc, char* argv[])
{

    //checking to see if we have image as an input
    if(argc == 3)
    {
        std::string fileDir = argv[1];

        cout << fileDir << endl;

        //initing the structure with the image from user input
        imageInit(fileDir, eNone);
    }
    else
    {
        cout << "\n======== ERROR ================================================================================" << endl;
        cout << "Invalid command line arguments received! Please enter the commands in the follow format:" << endl;
        cout << "./program.exe   < path/to/image >   <'CPU'|'CUDA'> \n" << endl;
        return 1;
    }

    std::string method = argv[2];   //getting convolution method input fromc cmd line

    //selecting convolve functions to be CPU or GPU based
    CImg<unsigned char>(*applyKernel)(const vector<vector<float>>& kernel, const CImg<unsigned char>&image);
        
    if (method == "CPU")    //cpu based convolution func
    {
        applyKernel = &applyKernelCPU;
    }
    else if (method == "CUDA")  //CUDA based convolution func
    {
        applyKernel = &applyKernelCUDA;
    }
    else {  //neither options

        cout << "Invalid option for processing method, please enter 'CPU' or 'CUDA' for the second argument" << endl;
        return 1;
    }


    // init menu by printing it to the user
    printMenu();
    moveToInputArea();


    bool running = true;
    while (running){

        clearInputArea();

		std::string input;
        cin >> input;

		// Check user input and call corresponding functions
		if (input == "1") {

            cout << "\033[41;1H";
            clearInputArea();
			cout << "Displaying Image..." << endl;
			dispImage();
			cout << "please close windows when ready..." << endl;

		} else if (input == "2") {

			//cout << "Setting Process to BoxBlur! " << endl;
			setProcess(eBoxBlur);
            setProcessGPU(eBoxBlur);
            cout << "\033[41;1H";
            clearInputArea();
			cout << "Process set to blur!";

		}
		else if (input == "3") {

			//cout << "Setting Process to Sharpen! " << endl;
			setProcess(eSharpen);
            setProcessGPU(eSharpen);
            cout << "\033[41;1H";
            clearInputArea();
			cout << "Process set to sharpen!";

		}
		else if (input == "4") {

            cout << "\033[41;1H";   //placing cursor in the right area
            clearInputArea();       //clearing feedback before pritning more...

            cout << "\033[41;1H";
            cout << "Please enter a kernal size for processing..." << endl;;

        kernelRetry:

			//getting input
			int kernelInput;
			cin >> kernelInput;

            moveToInputArea();
            clearInputArea();

			if (kernelInput % 2 == 0) {

				cout << "\033[41;1H";   //placing cursor in the right area
				clearInputArea();       //clearing feedback before pritning more...

				cout << "Please enter an odd number for a centered image kernel..." << endl;
				goto kernelRetry;
			}

            cout << "\033[41;1H";
            clearInputArea();

			setKernel(kernelInput);

            cout << "\033[41;1H";
            cout << "Process kernel set to: " << kernelInput << "x" << kernelInput << endl;
		}
		else if (input == "5") {

            cout << "\033[41;1H";
            clearInputArea();
			cout << "Beginning process..." << endl;

			if(getProcess() == eBoxBlur){
				boxBlur(applyKernel);
				cout << "Proessing complete!\nImage saved to root dir!\n";
				goto exit;
			}
			else if(getProcess() == eSharpen){

                //takes in a function pointer earlier to decide what convolution method it will use
				sharpen(applyKernel);
				cout << "Proessing complete!\nImage saved to root dir!\n";
				goto exit;

			}else{
				cout << "ERROR: NO PROCESS SELECTED. PLEASE MAKE A SELECTION BEFORE RUNNING PROCESS\n";
			}

		} else if (input == "6") {

			//std::cout << "Exiting program...\n";
			//goto exit;

		} else {
            cout << "\033[41;1H";
			cout << "Invalid input. Please try again.\n";
		}

        //cout << "done!" << endl;
        moveToInputArea();

        //end of running while loop 
    }

    // jump point for ending the program
    exit:
    return 0;
}




