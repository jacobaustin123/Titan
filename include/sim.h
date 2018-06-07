//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SIM_H
#define LOCH_SIM_H

#include "spring.h"
#include "mass.h"
#include "object.h"
#include "vec.h"

#include <algorithm>
#include <list>
#include <vector>
#include <set>

#ifdef GRAPHICS
#include "graphics.h"
#include "../src/common/shader.h"
#endif

static double G = 9.81;

struct Event {
    Event(void (*func)(), double time, double repeat = 0) {
        this -> func = func;
        this -> time = time;
        this -> repeat = repeat;
    }

    void (*func)();
    double time;
    double repeat;
};

struct compareEvents {
    bool operator()(const Event & a, const Event & b) {
        return a.time < b.time;
    }
};

class Simulation {
public:
    Simulation() { dt = 0; RUNNING = 0; } // constructors;
    ~Simulation();

    Mass * createMass(); // creat objects
    Spring * createSpring();
    Spring * createSpring(Mass * m1, Mass * m2, double k = 1.0, double len = 1.0);

    Plane * createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d
    Cube * createCube(const Vec & center, double side_length); // creates cube
    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    void setSpringConstant(double k);
    void defaultRestLength();
    void setMass(double m);
    void setMassDeltaT(double dt);

    Mass * getMass(int i) { return masses[i]; }
    Spring * getSpring(int i) { return springs[i]; }
    ContainerObject * getObject(int i) { return objs[i]; }

    void runFunc(void (*func)(), double time, double repeat = 0) {
        bpts.insert(Event(func, time, repeat));
    }

    void setBreakpoint(double time);

    void run(); // should set dt to min(mass dt) if not 0, resets everything
    void resume(); // same as above but w/out reset

    double time() { return T; }

    void printPositions();
    void printForces();

    double dt; // set to 0 by default, when run is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    int RUNNING;

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Constraint *> constraints;
    std::vector<ContainerObject *> objs;

    Mass * mass_arr;
    Spring * spring_arr;

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
#endif

    std::set<Event, compareEvents> bpts; // list of breakpoints

    void computeForces();

    Mass * massToArray();
    Spring * springToArray();
    void toArray();

    void massFromArray();
    void springFromArray();
    void fromArray();
};

#endif //LOCH_SIM_H
