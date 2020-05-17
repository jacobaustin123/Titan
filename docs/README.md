# Titan

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Linux](#linux-installation)
    - [Windows](#windows-installation)
    - [Troubleshooting](#troubleshooting)
- [API](#api-overview)
- [API Details](#api-details)
    - [Simulation](#simulation-methods)
    - [Mass](#mass-methods)
    - [Spring](#spring-methods)
    - [Container](#container-methods)
    - [Vec](#vec-methods)
- [Examples](#examples)
    - [Energy Conservation](#energy-conservation)
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

### Simulation

Titan runs simulations in a `Simulation` object which holds references to user defined objects, graphics buffers, constraints, and other data. Data is held on the GPU, with the user interacting with CPU objects which can fetch data from the GPU, modify it, and push it back to the simulation in real time. The Simulation object controls things like the duration of the run, graphics/rendering options like the viewport, and GPU parameters.

For example:

```cpp
Simulation sim;
Mass * m1 = sim.createMass(Vec(0, 0, 0));
Mass * m2 = sim.createMass(Vec(0, 0, 1));
Spring * s1 = sim.createSpring(m1, m2);

sim.setTimeStep(0.001);
sim.start();
```

### Mass
The simplest discrete simulation elements. Point masses have individual physical properties (mass, position, velocity, acceleration) and can be affected by forces exerted by different sources. 

```cpp
Simulation sim;
Mass * m1 = sim.createMass(Vec(0, 0, 0));
... // run simulation for a while
sim.getMass(m1); // pull updates from simulation
cout << m1 -> pos << endl; // position
cout << m1 -> vel << endl; // velocity
cout << m1 -> T << endl; // simulation time
```

### Spring

Springs connect pairs of masses together and apply simple Hooke's law forces. These forces are applied in parallel on the GPU, achieving a substantial performance improvement over a CPU based system.

```cpp
Simulation sim;
Mass * m1 = sim.createMass(Vec(0, 0, 0));
Mass * m2 = sim.createMass(Vec(0, 0, 1));
Spring * s1 = sim.createSpring(m1, m2);
... // run simulation for a while
cout << s1 -> _k << endl; // spring constant
cout << m1 -> _rest << endl; // rest length
cout << m1 -> _left -> pos << endl; // position of left mass
```

### Contaner

Masses and springs may belong to Containers which enable efficient and convenient transfers of information to and from the GPU. Once a mass or spring has been created, it can be added to a container of related objects. This container can the be pushed to or pulled from the GPU as a single unit, avoiding expensive copies and tedious boilerplate code.

```cpp
Simulation sim;
Lattice * l1 = sim.createLattice(titan::Vec(0, 0, 5), titan::Vec(4, 4, 4), 20, 20, 20); // container subbclass
l1 -> rotate(Vec(0, 0, 1), 3.14);
l1 -> translate(Vec(1, 2, 5));
l1 -> setMassValues(0.5); // set all masses in container to 0.5
```

### **Forces** 

Forces in Titan are defined by 3D vectors and affect masses during the simulation. Forces can be a result of several interactions:
* Mass-Spring Hooke’s forces due to Springs connecting masses
* Mass interactions with contact elements  
* Global accelerations (i.e. gravity) set up by the user

### Contacts
Contacts are predefined simulation elements that apply forces to masses when certain positional requirements are met. Contact elements only need to be initialized to start working.

Contacts included in Titan:

* **Plane:** Applies a normal force based on the masse’s displacement after breaching one face of the plane. 
* **Sphere:** Applies a normal force based on the masses’ displacement after breaching the sphere’s surface. 

```cpp
Simulation sim;
sim.createPlane(Vec(0, 0, 1), 0); // contact plane facing in the positive z direction with 0 offset
```

### Constraints

Constraints are positional limitations imposed on masses. Constraints need to be initialized and then associated to masses in order to work.

Constraints included in Titan:

* **Direction:** Constraints the movement of masses to one direction only. 
* **Plane:** Constraints the movement of masses to a plane. The plane is defined by a normal vector and and the masse’s position at the time of its application. 

Masses can also be marked as "fixed", meaning that they cannot move, and drag can be specified on individual masses, which will be applied according to a C v^2 law.

### Dynamic Simulations 

The Titan simulation environment is asynchronous and dynamic. The user can make arbitrary modifications to the simulation on the run, and these will be immediately reflected in the simulation. The user can:

* Fetch values from the GPU using sim.get(...)
* Push values to the GPU using sim.set(...)
* Add constraints and modify parameters of the simulation.

For example:

```cpp
Simulation sim;
Lattice * l1 = sim.createLattice(titan::Vec(0, 0, 5), titan::Vec(4, 4, 4), 20, 20, 20); // container subbclass
sim.start();
sim.wait(0.5); // sleep at 0.5 seconds
sim.get(l1); // pull updates from the GPU
l1 -> masses[0] -> pos = Vec(0, 0, 1); // set position of first mass.
sim.set(l1); // push updates to the GPU
```

## API Details

### Simulation Methods

```cpp
    Mass * createMass();
    Mass * createMass(const Vec & pos);

    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2);

    void deleteMass(Mass * m);
    void deleteSpring(Spring * s);

    void get(Mass *m); // pull objects from the GPU
    void get(Spring *s); // not really useful, no commands change springs
    void get(Container *c);

    void set(Mass * m); // push updates to the GPU
    void set(Spring *s);
    void set(Container * c);

    void getAll(); // get all objects from the GPU
    void setAll(); // set all objects

    // Global constraints (can be rendered)
    void createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    void createPlane(const Vec &abc, double d, double FRICTION_K, double FRICTION_S);  // creates half-space ax + by + cz < d

    void createBall(const Vec & center, double r ); // creates ball with radius r at position center
    void clearConstraints(); // clears global constraints only

    // Containers
    Container * createContainer();
    void deleteContainer(Container * c);

    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Beam * createBeam(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Container * importFromSTL(const std::string & path, double density = 10.0, int num_rays = 5); // density in vertices / volume

    // Bulk modifications, only update CPU
    void setAllSpringConstantValues(double k);
    void setAllMassValues(double m);
    void setTimeStep(double delta_t);
    void setGlobalAcceleration(const Vec & global_acc);
    void defaultRestLengths(); // makes all rest lengths equal to their current length

    // Control
    void start(); // start simulation

    void stop(); // stop simulation while paused, free all memory.
    void stop(double time); // stop simulation at time

    void pause(double t); // pause at time t, block CPU until t.
    void resume();

    void reset(); // reset the simulation
    
    void setBreakpoint(double time); // tell the program to stop at a fixed time (doesn't hang).

    void wait(double t); // wait fixed time without stopping simulation
    void waitUntil(double t); // wait until time without stopping simulation
    void waitForEvent();  // wait until event (e.g. breakpoint)

    double time();
    bool running();

    void printPositions();

    Spring * getSpringByIndex(int i);
    Mass * getMassByIndex(int i);
    Container * getContainerByIndex(int i);

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Container *> containers;

#ifdef GRAPHICS
    void setViewport(const Vec & camera_position, const Vec & target_location, const Vec & up_vector);
    void moveViewport(const Vec & displacement); // displace camera by vector
    glm::mat4 & getProjectionMatrix(); // access glm projection matrix
#endif
}
```

### Mass Methods

```cpp
class Mass {
    double m; // mass in kg
    double T; // local time of mass
    Vec pos; // position in m
    Vec vel; // velocity in m/s

    void setExternalForce(const Vec & v); // set external force applied every iteration
    Vec acceleration(); // get acceleration

    void addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num); // add constraint
    void clearConstraints(CONSTRAINT_TYPE type); // remove constraints of a certain type
    void clearConstraints(); // remove all constraints

    void setDrag(double C); // set drag with coefficient C
    void fix(); // make fixed (unable to move)
    void unfix(); // undo that
    
    Vec color; // RGB color
```

### Spring Methods

```cpp
enum SpringType {PASSIVE_SOFT, PASSIVE_STIFF, ACTIVE_CONTRACT_THEN_EXPAND, ACTIVE_EXPAND_THEN_CONTRACT};

class Spring {
public:
    Mass * _left; // pointer to left mass object // private
    Mass * _right; // pointer to right mass object

    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)

    SpringType _type; // type of spring
    double _omega; // frequency of oscillation
    double _damping; // damping on the spring

    void setRestLength(double rest_length); // set rest length to rest_length
    void defaultLength(); // sets rest length to distance between springs
    void changeType(SpringType type, double omega) { _type = type; _omega = omega;}
    void addDamping(double constant); // set damping coefficient

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right); //sets both right and left masses
}
```

### Container Methods

```cpp
class Container { // contains and manipulates groups of masses and springs
public:
    void translate(const Vec & displ); // translate all masses by fixed amount
    // rotate all masses around a fixed axis by a specified angle with respect to the center of mass in radians
    void rotate(const Vec & axis, double angle); 
    void setMassValues(double m); // set masses for all Mass objects
    void setSpringConstants(double k); // set k for all Spring objects
    void setRestLengths(double len); // set masses for all Mass objects

#ifdef CONSTRAINTS
    void addConstraint(CONSTRAINT_TYPE type, const Vec & v, double d);
    void clearConstraints();
#endif

    void fix(); // make all objects fixed
    void add(Mass * m); // add a mass
    void add(Spring * s); // add a spring
    void add(Container * c); // add another container

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
}
```

### Vec Methods

```cpp
class Vec {
public:
    Vec(double x, double y, double z); // initialization from x, y, and z values
    Vec(const std::vector<double> & v);
    Vec & operator+=(const Vec & v);
    Vec & operator-=(const Vec & v);
    CUDA_DEVICE void atomicVecAdd(const Vec & v);
    Vec operator-() const;
    double & operator [] (int n);
    const double & operator [] (int n) const;
    friend Vec operator+(const Vec & v1, const Vec & v2);
    friend Vec operator-(const Vec & v1, const Vec & v2);
    friend Vec operator*(const double x, const Vec & v);
    friend Vec operator*(const Vec & v, const double x);
    friend bool operator==(const Vec & v1, const Vec & v2);
    friend Vec operator*(const Vec & v1, const Vec & v2);
    friend Vec operator/(const Vec & v, const double x);
    friend Vec operator/(const Vec & v1, const Vec & v2);
    friend std::ostream & operator << (std::ostream & strm, const Vec & v);

    void print(); // supports CUDA printing
    double norm() const;
    double sum() const;
    double dot(const Vec & a, const Vec & b);
    Vec cross(const Vec &v1, const Vec &v2); // cross product
};
```

## Examples

### Energy conservation

```cpp
#include <Titan/sim.h>

int main() {
    titan::Simulation sim;
    sim.createLattice(titan::Vec(0, 0, 5), titan::Vec(4, 4, 4), 20, 20, 20);
    
    sim.setAllSpringConstantValues(100);
    sim.setTimeStep(0.0001);
    sim.setGlobalAcceleration(titan::Vec(0, 0, -9.8));
    sim.defaultRestLengths();

    sim.createPlane(titan::Vec(0, 0, 1), 0);
    sim.start();

    double total_energy = titan::test::energy(sim);
    double avg_energy = total_energy;
    double alpha = 0.7;
    while (sim.time() < 5) {
        sim.wait(0.1);
        avg_energy = (1 - alpha) * titan::test::energy(sim) + alpha * avg_energy;
        EXPECT_NEAR(avg_energy, total_energy, total_energy * tol);

        sim.resume();
    }

    sim.stop();
}
```

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

