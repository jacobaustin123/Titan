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


class Simulation {
public:
    // Create
    pyMass createMass();
    pyMass createMass(const Vec & pos);

    pySpring createSpring();
    pySpring createSpring(pyMass m1, pyMass m2);

    // Delete
    void deleteMass(Mass * m);
    void deleteSpring(Spring * s);
    void deleteContainer(Container * c);

    void get(Mass *m);
    void get(Spring *s); // not really useful, no commands change springs
    void get(Container *c);

    void set(Mass * m);
    void set(Spring *s);
    void set(Container * c);

    void getAll();
    void setAll();

    // Constraints
    void createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    void createBall(const Vec & center, double r ); // creates ball with radius r at position center

    void clearConstraints(); // clears global constraints only

    // Containers
    Container * createContainer();
    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    // Bulk modifications, only update CPU
    void setSpringConstant(double k);
    void setMassValues(double m);
    void setDeltaT(double dt);

    void defaultRestLength();

    // Control
    void start(double time = 1E20); // start simulation, run until simulation time T

    void stop(); // stop simulation while paused, free all memory.
    void stop(double time); // stop simulation at time

    void pause(double t); // pause at time t
    void resume();

    void wait(double t); // wait fixed time without stopping

    double time() { return T; }
    double running() { return RUNNING; }

    void printPositions();
    void printForces();

    Simulation();
    ~Simulation();

    Spring * getSpringByIndex(int i);
    Mass * getMassByIndex(int i);
    Container * getContainerByIndex(int i);

private:

    void freeGPU();

    void setBreakpoint(double time);
    void _run();

    void waitUntil(double t);
    void waitForEvent();

    void execute(); // same as above but w/out reset

    //Prints
    void printSprings();

    pyMass createMass(Mass * m); // utility
    pySpring createSpring(Spring * s); // utility

    double stop_time;

    double dt; // set to 0 by default, when start is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    bool RUNNING;
    bool STARTED;
    bool ENDED;
    bool FREED;

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Constraint *> constraints;
    std::vector<Container *> objs;

    thrust::device_vector<CUDA_MASS *> d_masses;
    thrust::device_vector<CUDA_SPRING *> d_springs;

    thrust::device_vector<CudaContactPlane> d_planes; // used for constraints
    thrust::device_vector<CudaBall> d_balls; // used for constraints

    CUDA_GLOBAL_CONSTRAINTS d_constraints;
    bool update_constraints;

    void updateCudaParameters();

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

__global__ void createSpringPointers(CUDA_SPRING ** ptrs, CUDA_SPRING * data, int size);
__global__ void createMassPointers(CUDA_MASS ** ptrs, CUDA_MASS * data, int size);

__global__ void computeSpringForces(CUDA_SPRING * device_springs, int num_springs);
//__global__ void computeMassForces(CUDA_MASS * device_masses, int num_masses);
__global__ void massForcesAndUpdate(CUDA_SPRING * device_springs, Constraint ** d_constraints, int num_springs, int num_constraints);
//__global__ void update(CUDA_MASS * d_mass, int num_masses);

#endif //LOCH_SIM_H
