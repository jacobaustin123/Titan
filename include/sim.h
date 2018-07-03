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
#define THREADS_PER_BLOCK 1024

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
#include <thread>

static double G = 9.81;


struct AllConstraints {
    CUDA_PLANE * d_planes;
    CUDA_BALL * d_balls;

    int num_planes;
    int num_balls;
};

class Simulation {
public:
    Simulation() {
        dt = 0;
        RUNNING = false;
        STARTED = false;
        update_constraints = true;

#ifdef GRAPHICS
        resize_buffers = true;
        update_colors = true;
        update_indices = true;

        lineWidth = 1;
        pointSize = 3;
#endif
    }

    ~Simulation();

    // Create
    Mass * createMass();
    Mass * createMass(const Vec & pos);

    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2);

    // Delete
    void deleteMass(Mass * m);
    void deleteSpring(Spring * s);
    void deleteContainer(Container * c);

    // Constraints
    Plane * createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    Ball * createBall(const Vec & center, double r ); // creates ball with radius r at position center

    // Containers
    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    // Bulk modifications
    void setSpringConstant(double k);
    void defaultRestLength();
    void setMass(double m);
    void setMassDeltaT(double dt);

    // Control
    void start(); // start simulation
    void pause(double t); // pause at time t
    void resume();

    void wait(double t); // wait fixed time without stopping

    double time() { return T; }
    double running() { return RUNNING; }



    // private

    void setBreakpoint(double time);
    void _run();

    void waitUntil(double t);
    void waitForEvent();

    void execute(); // same as above but w/out reset

    //Prints
    void printPositions();
    void printForces();
    void printSprings();
//    void printSpringForces();

    Mass * createMass(Mass * m); // utility
    Spring * createSpring(Spring * s); // utility


    double dt; // set to 0 by default, when start is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    bool RUNNING;
    bool STARTED;

    std::list<Mass *> masses;
    std::list<Spring *> springs;
    std::list<Constraint *> constraints;
    std::list<Container *> objs;

    thrust::device_vector<CUDA_MASS *> d_masses;
    thrust::device_vector<CUDA_SPRING *> d_springs;

    AllConstraints d_constraints;
    bool update_constraints;

    std::set<double> bpts; // list of breakpoints

    CUDA_MASS ** d_mass;
    CUDA_SPRING ** d_spring;

    int massBlocksPerGrid;
    int springBlocksPerGrid;

    CUDA_MASS ** massToArray();
    CUDA_SPRING ** springToArray();
    void constraintsToArray();
    void toArray();

    void massFromArray();
    void springFromArray();
    void constraintsFromArray();
    void fromArray();

    std::thread gpu_thread;

#ifdef GRAPHICS

#ifdef SDL2
    SDL_Window * window;
    SDL_GLContext context;
    void createSDLWindow();
#else
    GLFWwindow * window;
    void createGLFWWindow();
#endif

    GLuint VertexArrayID;
    GLuint programID;
    GLuint MatrixID;
    glm::mat4 MVP;

    GLuint vertices;
    GLuint colors;
    GLuint indices;

    void clearScreen();
    void renderScreen();
    void updateBuffers();
    void generateBuffers();
    void resizeBuffers();
    void draw();

    bool update_indices;
    bool update_colors;
    bool resize_buffers;

    int lineWidth;
    int pointSize;
#endif
};

#ifdef GRAPHICS
#ifndef SDL2
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
#endif
#endif

__global__ void computeSpringForces(CUDA_SPRING * device_springs, int num_springs);
__global__ void computeMassForces(CUDA_MASS * device_masses, int num_masses);
__global__ void massForcesAndUpdate(CUDA_SPRING * device_springs, Constraint ** d_constraints, int num_springs, int num_constraints);
__global__ void update(CUDA_MASS * d_mass, int num_masses);

#endif //LOCH_SIM_H
