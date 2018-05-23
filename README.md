# Loch
CUDA-based physics simulation sandbox using springs and masses to simulate flexible robots and other mechanical objects. Built in C++ but with a future Python wrapper.

## To compile

Create a build directory inside the main directory, navigate to it, and the run ```cmake ..``` and then ```make```. This will produce a ```main``` executable which can be run using ```./main```.

## Some temporary notes for compiling using CLion

For CLion to compile and run, you need to set the working directory to the src folder. Go to Run/Edit Configurations and set the working directory to src.

Right now there are two branches, the master branch and the graphics branch. The graphics branch supports a very badly-performant OpenGL graphics rendering for certain objects (mainly just what has been hardcoded in the main function). This branch requires the glfw, glm, OpenGL, and GLEW libraries. 

These can be installed on Linux with sudo apt-get glfw3 glm glew. On Mac OS, use homebrew to install them. For Windows, we're working on it.
