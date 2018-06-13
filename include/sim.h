//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SIM_H
#define LOCH_SIM_H

#include "spring.h"
#include "mass.h"
#include "object.h"
#include "vec.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAX_BLOCKS 65535 // max number of CUDA blocks

#ifndef GRAPHICS
#define NUM_QUEUED_KERNELS 4 // number of kernels to queue at a given time (this will reduce the frequency of updates from the CPU by this factor
#endif

#ifdef GRAPHICS

#include "graphics.h"
#include "common/shader.h"
#include <cuda_gl_interop.h>

#endif

#include <algorithm>
#include <list>
#include <vector>
#include <set>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>

static double G = 9.81;

class Simulation {
public:
    Simulation() {
        dt = 0;
        RUNNING = 0;

#ifdef GRAPHICS
        update_colors = true;
        update_indices = true;
        lineWidth = 1;
        pointSize = 10;
#endif
    }

    ~Simulation();

    //Create
    Mass * createMass();
    Mass * createMass(const Vec & pos);

    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2, double k = 1.0, double len = 1.0);

    Plane * createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Ball * createBall(const Vec & center, double r ); // creates ball with radius r at position center

    void setSpringConstant(double k);
    void defaultRestLength();
    void setMass(double m);
    void setMassDeltaT(double dt);

    void setBreakpoint(double time);

    //Control
    void run(); // should set dt to min(mass dt) if not 0, resets everything
    void resume(); // same as above but w/out reset

    //Get
    double time() { return T; }

    //Prints
    void printPositions();
    void printForces();
//    void printSprings();
//    void printSpringForces();


// private from here
    double dt; // set to 0 by default, when run is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    int RUNNING;

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Constraint *> constraints;
    std::vector<ContainerObject *> objs;

    thrust::device_vector<Constraint> d_constraints;

    CUDA_MASS * d_mass;
    CUDA_SPRING * d_spring;

    std::set<double> bpts; // list of breakpoints

    CUDA_MASS * massToArray();
    CUDA_SPRING * springToArray();
    Constraint * constraintsToArray();
    void toArray();

    void massFromArray();
    void springFromArray();
    void constraintsFromArray();
    void fromArray();

#ifdef GRAPHICS

    GLuint VertexArrayID;
    GLuint programID;
    GLuint MatrixID;
    GLFWwindow * window;
    glm::mat4 MVP;

    GLuint vertices;
    GLuint colors;
    GLuint indices;

    void clearScreen();
    void renderScreen();
    void updateBuffers();
    void generateBuffers();
    void draw();

    bool update_indices;
    bool update_colors;

    int lineWidth;
    int pointSize;
#endif
};

__global__ void computeSpringForces(CUDA_SPRING * device_springs, int num_springs);
__global__ void computeMassForces(CUDA_MASS * device_masses, int num_masses);
__global__ void massForcesAndUpdate(CUDA_SPRING * device_springs, int num_springs);
__global__ void update(CUDA_MASS * d_mass, int num_masses);

#endif //LOCH_SIM_H
