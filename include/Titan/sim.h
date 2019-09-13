//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_SIM_H
#define TITAN_SIM_H

#include "spring.h"
#include "mass.h"
#include "object.h"
#include "vec.h"

#define MAX_BLOCKS 65535 // max number of CUDA blocks
#define THREADS_PER_BLOCK 1024

#ifndef GRAPHICS
#define NUM_QUEUED_KERNELS 4 // number of kernels to queue at a given time (this will reduce the frequency of updates from the CPU by this factor
#endif

#ifdef GRAPHICS

#include "graphics.h"
#include "shader.h"

#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <list>
#include <vector>
#include <set>
#include <thread>

class Simulation {
public:
    // Create
    Mass * createMass();
    Mass * createMass(const Vec & pos);

    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2);

    // Delete
    void deleteMass(Mass * m);
    void deleteSpring(Spring * s);

    void get(Mass *m);
    void get(Spring *s); // not really useful, no commands change springs
    void get(Container *c);

    void set(Mass * m);
    void set(Spring *s);
    void set(Container * c);

    void getAll();
    void setAll();

    // Global constraints (can be rendered)
    void createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    void createBall(const Vec & center, double r ); // creates ball with radius r at position center

    void clearConstraints(); // clears global constraints only

    // Containers
    Container * createContainer();
    void deleteContainer(Container * c);

    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Robot * createRobot(const Vec & center, const cppn& encoding, double side_length,  double omega=1.0, double k_soft=2e3, double k_stiff=2e5);
    Beam * createBeam(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);
    Container * importFromSTL(const std::string & path, double density = 10.0, int num_rays = 5); // density is vertices / volume

    // Bulk modifications, only update CPU
    void setAllSpringConstantValues(double k);
    void setAllMassValues(double m);
    void setAllDeltaTValues(double dt);

    void defaultRestLength();

    // Control
    void start(); // start simulation

    void stop(); // stop simulation while paused, free all memory.
    void stop(double time); // stop simulation at time

    void pause(double t); // pause at time t
    void resume();

    void reset(); // reset the simulation
    
    void setBreakpoint(double time); // tell the program to stop at a fixed time (doesn't hang).

    void wait(double t); // wait fixed time without stopping
    void waitUntil(double t);
    void waitForEvent();

    double time();
    bool running();

    void printPositions();
    void printForces();

    Simulation();
    ~Simulation();

    Spring * getSpringByIndex(int i);
    Mass * getMassByIndex(int i);
    Container * getContainerByIndex(int i);

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Container *> containers;

    void setGlobalAcceleration(const Vec & global);

#ifdef GRAPHICS
    void setViewport(const Vec & camera_position, const Vec & target_location, const Vec & up_vector);
    void moveViewport(const Vec & displacement);
#endif

private:
    void freeGPU();
    void _run();

    void execute(); // same as above but w/out reset

    //Prints
    void printSprings();

    Mass * createMass(Mass * m); // utility
    Spring * createSpring(Spring * s); // utility

    double dt; // set to 0 by default, when start is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    static bool RUNNING;
    static bool STARTED;
    static bool ENDED;
    static bool FREED;
    static bool GPU_DONE;

    std::vector<Constraint *> constraints;

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
    Vec global; // global force

#ifdef GRAPHICS

#ifdef SDL2
    static SDL_Window * window;
    static SDL_GLContext context;
    void createSDLWindow();
#else
    static GLFWwindow * window;
    void createGLFWWindow();
#endif

    static GLuint VertexArrayID;
    static GLuint programID;
    static GLuint MatrixID;
    static glm::mat4 MVP;

    static GLuint vertices;
    static GLuint colors;
    static GLuint indices;

    void clearScreen();
    void renderScreen();
    void updateBuffers();
    void generateBuffers();
    void resizeBuffers();
    void draw();

    static bool update_indices;
    static bool update_colors;
    static bool resize_buffers;

    static int lineWidth;
    static int pointSize;

    static Vec camera;
    static Vec looks_at;
    static Vec up;
#endif
};

#endif //TITAN_SIM_H
