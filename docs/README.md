# Titan

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Linux](#linux-installation)
    - [Windows](#windows-installation)
    - [Troubleshooting](#troubleshooting)
- [API](#api)
- [About](#about)

## Overview

**Titan** is a versatile CUDA-based physics simulation library that provides a GPU-accelerated environment for physics primatives like springs and masses. Library users can create masses, springs, and more complicated objects, apply constraints, and modify simulation parameters in real time, while the simulation runs asynchronously on the GPU. Titan has been used for GPU-accelerated reinforcement learning, physics and biomedical research, and topology optimization.

```C++
#include <Titan/sim.h>

int main() {
  titan::Simulation sim;
  sim.createLattice(titan::Vec(0, 0, 10), titan::Vec(5, 5, 5), 5, 5, 5); // create lattice with center at (0, 0, 10) and given dimensions
  sim.createPlane(titan::Vec(0, 0, 1), 0); // create constraint plane
  sim.start();
}
```

<img src="https://i.imgur.com/zdB0ZPg.gif" width="400" height="400">

For more examples and troubleshooting, see the [github wiki](https://github.com/jacobaustin123/Titan/wiki/Set-Up). We also have a user [Google Group](https://groups.google.com/forum/#!forum/titan-library) where you can ask questions about installation and usage, and make feature requests.

Also see [this overview video](https://www.youtube.com/watch?v=IvZNL8jd77s) for an overview of the library and its capabilities.

## Installation

### Requirements

#### Windows
* [Microsoft Visual Studio 2015](https://msdn.microsoft.com/en-us/library/e2h7fzkw.aspx) or [2017](https://visualstudio.microsoft.com/downloads/)
* [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
* [Git](https://git-scm.com/downloads)
* [vcpkg](https://github.com/Microsoft/vcpkg) for installing dependencies

#### Linux

* gcc compiler
* [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
* [Git](https://git-scm.com/downloads)

### Linux Installation

#### Linux Quick Installation:

TLDR (note this requires CUDA to be installed):

```bash
./setup.sh
./clean-build.sh debug # (or release)
```

The `clean-build.sh` script can also take CMake flags as arguments. For example, to build tests, do 

```bash
./setup.sh
./clean-build.sh debug --DTITAN_BUILD_TESTS=ON
```

or to build with support for Verlet integration do

```bash
./setup.sh
./clean-build.sh debug --DTITAN_BUILD_TESTS=ON -DVERLET=ON
```

Other options include `-DGRAPHICS`, `-DCONSTRAINTS`, `-DRK2`. See the CMakeFile for full documentation.

If this doesn't work, try the following manual installation steps:

#### 1. Install the NVIDIA CUDA Toolkit

Download the NVIDIA CUDA Toolkit from this [link](https://developer.nvidia.com/cuda-downloads) and follow the quick install instructions. If the installation fails, try again using the advanced installation tab after unchecking Visual Studio Integration. This is a known CUDA big caused by incompatibilities with some Visual Studio versions.

#### 2. Install vcpkg

We will be using the vcpkg utility to handle dependencies for this project. The library optionally supports graphics rendering of the mass spring simulation using OpenGL, with utilities provided by the GLM, GLEW, and GLFW libraries. These libraries can be installed in any fashion, but the Microsoft vcpkg package manager provides a convenient method. To use vcpkg,

1. Go to the vcpkg [GitHub](https://github.com/Microsoft/vcpkg) and clone the repository into your user account (ideally in /home/username/Documents or /home/username/Downloads) using the following:

``` 
cd ~/Documents
git clone https://github.com/Microsoft/vcpkg.git
```

2. Follow the installation/setup instructions provided in the GitHub (reproduced here) to build vcpkg

```
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
```

The last command will output a path to the vcpkg cmake file which you will need to include in future projects to use the Titan library. Save this output, for example: "-DCMAKE_TOOLCHAIN_FILE=/home/username/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake"

#### 3. Download and Install Titan

To install the library using vcpkg, clone the github repository to a folder on your computer using

```
git clone https://github.com/jacobaustin123/Titan.git
```

Inside the newly downloaded Titan directory, navigate to Titan/vcpkg and copy the "titan" directory there to the ports folder in the vcpkg installation folder from step 2. For example, if vcpkg was installed in C:/vcpkg, run:

```
cd Titan/vcpkg
cp -r titan ~/Documents/vcpkg/ports/
```

Then in the vcpkg installation folder, run

```
./vcpkg install titan --head
```

which will handle all of the dependencies for you. If vcpkg fails to find CUDA, try running ```export CUDA_PATH=/usr/local/cuda```, or whatever the path is to your CUDA installation. You can copy that line into your .bashrc file to avoid having to run it every time. At the moment, due to an issue with CUDA and CMake, you will need to include the line 

```cmake
project(myproject LANGUAGES CXX CUDA)
```

at the beginning of whatever project uses the Titan library, with the myproject variable replaced by the name of your project. This is because certain environment variables which are needed by the library are not being set properly. In some cases, if CMake cannot find CUDA, you will need to manually set

```cmake
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
```
or whatever the path to the CUDA nvcc compiler is.


### Windows Installation:

This quick installation guide assumes you already have a C++ compiler installed, like Microsoft Visual Studio 2015/2017. We will:

* Install the NVIDIA CUDA Toolkit
* Install the Microsoft vcpkg package manager
* Build and install Titan and its dependencies

#### 1. Install the NVIDIA CUDA Toolkit

Download the NVIDIA CUDA Toolkit from this [link](https://developer.nvidia.com/cuda-downloads) and follow the quick install instructions. If the installation fails, try again using the advanced installation tab after unchecking Visual Studio Integration. This is a known CUDA big caused by incompatibilities with some Visual Studio versions.

#### 2. Install vcpkg

We will be using the vcpkg utility to handle dependencies for this project. The library optionally supports graphics rendering of the mass spring simulation using OpenGL, with utilities provided by the GLM, GLEW, and GLFW libraries. These libraries can be installed in any fashion, but the Microsoft vcpkg package manager provides a convenient method. To use vcpkg,

1. Go to the vcpkg [GitHub](https://github.com/Microsoft/vcpkg) and clone the repository into your user account (ideally in C:/vcpkg or C:/Users/.../Documents) using the following:

``` 
cd C:/
git clone https://github.com/Microsoft/vcpkg.git
```

2. Then follow the installation/setup instructions provided in the GitHub (reproduced here) to build vcpkg

```
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install
```

The last command will output a path to the vcpkg cmake file which you will need to include in future projects to use the Titan library. Save this output, for example: "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

#### 3. Download and Install Titan

To install the library using vcpkg, clone the github repository to a folder on your computer using

```
git clone https://github.com/jacobaustin123/Titan.git
```

Inside the newly downloaded Titan directory, navigate to Titan/vcpkg and copy the "titan" directory there to the ports folder in the vcpkg installation folder from step 2. For example, if vcpkg was installed in C:/vcpkg, run:

```
cd Titan/vcpkg
cp -r titan C:/vcpkg/ports/titan
```

Then in the vcpkg installation folder, run

```
./vcpkg install --triplet x64-windows titan --head
```

This will download and install all the necessary dependencies into the vcpkg install folder. Everything is now installed, and you can use it to build a sample project. See the [next](https://github.com/jacobaustin123/Titan/wiki/Creating-a-New-Project) page for instructions on using the library with your projects. Note that the vcpkg output you saved earlier will need to be passed to any CMake project you use the library with.


### Troubleshooting

#### Using an IDE with the Titan library and vcpkg

Vcpkg is a convenient cross-platform dependency management system developed by Microsoft, designed to work seamlessly with CMake. In any of the installation instructions above, you saved a command like "-DCMAKE_TOOLCHAIN_FILE=/Users/username/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake". This is the command that must be passed to CMake to build a project using dependencies installed by vcpkg. This can be passed directly to vcpkg on the command-line, and for IDEs it can be included as directed below:

#### Using Visual Studio with Titan

To make Visual Studio compatible with Titan and CUDA, you may need to make the following changes:

1. In Visual Studio, go to CMake/CMake Settings and generate a CMakeSettings.json file for your project. In this file, under the x64-debug and x64-release targets, you may need to add the variables section of the following example. 

```json
{
      "name": "x64-Release",
      ...,
      "variables": [
             {
               "name": "CMAKE_TOOLCHAIN_FILE",
               "value": "${env.VCPKG_DIR}"
             }
     ]
}
```

If this fails, change env.VCPKG_DIR to the actual path to the vcpkg directory. 

2. To work with CUDA, you may need to:

Open C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt\host_config.h
Find the line  #if _MSC_VER < 1600 || _MSC_VER > 1913 and replace it with just  #if _MSC_VER < 1600

3. If you are unable to get Visual Studio 2017 to work, use Visual Studio 2015, but manually copy the Titan.dll dynamic library from the vcpkg/installed directory into your project directory. Then you should be able to include the headers found in the vcpkg/installed directory and link to the library file. 

#### Using Microsoft Visual Studio 2015 with Titan

To install and use Microsoft Visual Studio Community 2015 with Titan, download Visual Studio Community 2015 (with Update 3) from the provided [link](https://msdn.microsoft.com/en-us/library/e2h7fzkw.aspx) and follow the installer instructions to install the Visual C++ compiler and v140 toolkit. You do not need to install the Visual Studio IDE or any other tools (only Visual C++ tools). You may need to subscribe for free to My Visual Studio to access older versions.

#### Using Microsoft Visual Studio 2017 with Titan

To install and use Microsoft Visual Studio Community 2017 with Titan, download Microsoft Visual Studio Community 2017 from Microsoft [(link)](https://visualstudio.microsoft.com/downloads/) and follow the installer instructions to install the Visual C++ compiler and v140 toolkit. You do not need to install the Visual Studio IDE or any other tools (only Visual C++ tools), although you may want to install the IDE if you prefer to develop in Visual Studio over CLion.

_Once Visual Studio 2017 is installed, you will have to make a few changes to let it interface with CUDA._ For some reason, CUDA is made incompatible with newer versions of Visual Studio, but disabling this has no adverse effects. To do this, you may need to 

* Open C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt\host_config.h
* Find the line ```#if _MSC_VER < 1600 || _MSC_VER > 1913``` and replace it with just  ```#if _MSC_VER < 1600```

#### Using CLion with Titan

CLion is a cross-platform IDE developed by IntelliJ, the creators of PyCharm and IDEA. The IDE uses CMake by default, so it is ideal for including our CMake project. To build and run your project with CLion, several settings changes need to be made.

1. First, in Settings/Build, Execution, Deployment/CMake, make sure CMake Options includes the command

```
-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
``` 

where the path points to the vcpkg folder. The above is an example for Windows. On Unix systems it will look like "-DCMAKE_TOOLCHAIN_FILE=/Users/username/Documents/vcpkg/scripts/buildsystems/vcpkg.cmake". If you have vcpkg installed in a different directory, use that path instead. This is the path you saved earlier in the process.

2. Make sure the compiler in Settings/Build, Execution, Deployment/Toolchains is set to Visual Studio 2015 or 2017 (14.0), and the architecture is set to 64-bit (amd64 on Windows). _Do not use amd64_arm or any other compiler option_.

* Note that there is sometimes a bug in CLion with CUDA support that causes it to run the wrong executable - if CLion is unable to run an executable, manually run the executable found in the project directory (it will be found in the cmake-build-debug or cmake-build-release folder depending on your settings). 

* Note that CLion will sometimes find the library installed in the wrong directory, not the vcpkg version. If CLion is unable to find the library, or seems to have found the wrong version, try navigating to Program Files and Program Files (x86) and deleting any folder called "Titan". Then reload the CMake project (using File/Reload CMake Project). Sometimes you will also need to delete the cmake-build-debug folder, close CLion, and then reopen it and run File/Reload CMake Project.

#### Uninstalling

To remove the titan library on Windows, simply run

```
./vcpkg remove titan --triplet x64-windows
```

and on Unix-based systems, run

```
./vcpkg remove titan
```

It can be reinstalled at any time using ```./vcpkg install titan``` or ```./vcpkg install titan --triplet x64-windows```.

## API Overview

Lorem Ipsum Ipsum Lorem

## About

This software was written by Jacob Austin and Rafael Corrales Fatou as part of a project led by Professor Hod Lipson at the Creative Machines Lab at Columbia University. This software is released under an Apache 2.0 license.

If using this software in published work, please cite

```
J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
```

or use the BibTex

```
@inproceedings {TitanICRA,
   title	= {Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA},
   author	= {Jacob Austin, Raphael Corrales-Fatou, Soia Wyetzner, and Hod Lipson},
   bookTitle	= {Proc. of the {IEEE} International Conference on Robotics and Automation},
   month	= {May},
   year		= {2020},
   location	= {Paris, France}
}
```

