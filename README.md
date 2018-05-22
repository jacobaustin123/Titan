# Loch
acc CUDA-based physics simulation sandbox using springs and masses to simulate flexible robots and other mechanical objects. Built in C++ but with a Python wrapper.

## Some temporary notes for compiling using CLion

For CLion to compile and run, you need to set the working directory to the src folder. Go to Run/Edit Configurations and set the working directory to src.

Right now the code depends on the glfw, glm, and GLEW libraries in addition to OpenGL. These can be installed on Linux with sudo apt-get glfw3 glm glew. On Mac OS, use homebrew to install them. For Windows, we're working on it. The required Windows libraries are included in the src/dependencies folder, but the CMake configuration isn't working yet.
