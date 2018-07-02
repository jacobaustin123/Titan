# Loch
A CUDA-based physics simulation sandbox written in C++ which uses springs and masses to simulate flexible robots and other mechanical objects. 

## To compile and run

The project currently has two branches, the master branch (CPU based) and the CUDA branch (GPU based). While both branches have graphics-free executables, they currently still depend on graphics packages, namely glfw3, glm, and GLEW. On Mac OS and Linux, dependencies should be handled using homebrew and apt-get respectively. For Mac OS, run

```$ brew install glm glfw3 GLEW```

and for Linux run

```$ sudo apt-get install glm glfw3 GLEW```

On Windows, these dependencies should be installed using Microsoft vcpkg. To set up vcpkg, run the following from Microsoft Powershell.

```PS> cd ~
PS> mkdir tools
PS> cd tools
PS> git clone https://github.com/Microsoft/vcpkg.git
PS> cd vcpkg
PS> .\bootstrap-vckpg.bat
PS> .\vcpkg integrate install # Keep the output showing `CMAKE_TOOLCHAIN_FILE` variable
PS> .\vcpkg integrate powershell # You may need to 
PS> Set-ExecutionPolicy Unrestricted -Scope CurrentUser # May need to run this to allow the vcpkg powershell integration to work
```

Then to install the dependencies run

```PS>./vcpkg --triplet x64-windows install glfw3 GLEW glm```

from the vcpkg directory. Then build and install Loch as follows:

```$ git clone https://github.com/ja3067/Loch.git
$ cd Loch
$ mkdir build
$ cd build
$ cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg root]/scripts/buildsystems/vcpkg.cmake
$ make
$ ./graphics
```

For the CUDA branch, CUDA 9.2 also must be installed, as well as the Windows VS compiler. The CUDAgraphics branch has the same requirements as above.

## Troubleshooting

### Using with CLion

To build and run with CLion, several settings changes need to be made. First, in Settings/Build, Execution, Deployment/CMake, make sure CMake Options is set to "-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg root]/scripts/buildsystems/vcpkg.cmake" (no quotes). Also, make sure the compiler in Settings/Build, Execution, Deployment/Toolchains is set to Visual Studio 2015 or 2017 (14.0). 

Also, to run in CLion, you need to edit the configuration in Run/Edit Configurations by setting the working directory to the \[Loch root\]/src. Note that there is a bug in CLion with CUDA support that causes it to run the wrong executable - if CLion is unable to start the executable, manually run the graphics or nographics executables in .../Loch/cmake-build-debug. If the program is unable to find the shaders, manually copy the Loch/src/shaders directory to cmake-build-debug.
