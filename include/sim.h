//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SIM_H
#define LOCH_SIM_H

#include "spring.h"
#include "mass.h"
#include "object.h"
#include "vec.h"

#include <list>
#include <vector>
#include <set>

static double G = 9.81;

//Ball & createBall(double radius = 1.0, const Vec & center = Vec(0, 0, 0));
//Plane & createPlane(const Vec & abc = Vec(0, 0, 0), double d = 0); // creates half-space ax + by + cz < d

//Mass * getMassByIndex(int n); // can support negative values (from end)
//Spring * getSpringByIndex(int n);

//void resume();

// std::vector<Constraint *> constraints; // global constraints, for initial example

class Simulation {
public:
    Simulation() { dt = 0; RUNNING = 0; }
    ~Simulation();

    Mass * createMass();
    Spring * createSpring();

    void setBreakpoint(double time);

    void run(); // should set dt to min(mass dt) if not 0, resets everything
    void resume(); // same as above but w/out reset

    double time() { return T; }

    Plane * createPlane(const Vec & abc, double d ); // creates half-space ax + by + cz < d

    Cube * createCube(const Vec & center, double side_length); // creates half-space ax + by + cz < d

    void printPositions();
    void printForces();

private:
    double dt; // set to 0 by default, when run is called will be set to min(mass dt) unless previously set
    double T; // global simulation time

    int RUNNING;

    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
    std::vector<Constraint *> constraints;
    std::vector<ContainerObject *> objs;

    Mass * mass_arr;
    Spring * spring_arr;

    std::set<double> bpts; // list of breakpoints

    void computeForces();

    Mass * massToArray();
    Spring * springToArray();
    void toArray();

    void massFromArray();
    void springFromArray();
    void fromArray();
};

#endif //LOCH_SIM_H
