# Loch
CUDA-based physics simulation sandbox using springs and masses to simulate flexible robots and other mechanical objects. Built in C++ but with a future Python wrapper.

## To compile and run

The project has 3 main branches right now - the master/newAPI branch, the graphics/newAPIgraphics branch, and the CUDA branch. Each has their own dependency requirements. 

The master branch has no dependencies, and can be compiled and run using cmake by creating a build directory inside the repo, and running ```cmake ..``` and then ```make``` from that folder. This will produce a ```main``` executable which can be run using ```./main```.

The graphics branch requires OpenGL, glm, GLEW, and glfw3, which can be installed using vcpkg, apt-get, or brew on Windows/Linux/Mac OS respectively. To use vcpkg, run

```S> cd ~
PS> mkdir tools
PS> cd tools
PS> git clone https://github.com/Microsoft/vcpkg.git
PS> cd vcpkg
PS> .\bootstrap-vckpg.bat
PS> .\vcpkg integrate install # Keep the output showing `CMAKE_TOOLCHAIN_FILE` variable
PS> .\vcpkg integrate powershell # You may need to 
PS> Set-ExecutionPolicy Unrestricted -Scope CurrentUser # May need to run this to allow the vcpkg powershell integration to work

```

to install vcpkg, and then run

```PS> vcpkg --triplet x64-windows install glfw3 GLEW glm```

then when compiling with CMake from the build directory, run CMake using ```cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake```

For the CUDA branch (and future CUDA/graphics branch), CUDA 9.2 also must be installed, as well as the Windows VS compiler. The CUDAgraphics branch has the same requirements as above.
