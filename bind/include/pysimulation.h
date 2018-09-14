//
// Created by rcorr on 9/13/2018.
//

#ifndef LOCH_PYSIMULATION_H
#define LOCH_PYSIMULATION_H

#include "sim.h"
#include "pymass.h"
#include "pyspring.h"

class pySimulation{
public:
    //constructor
    pySimulation() = default;
//    ~pySimulation();

    Simulation sim;

    //Create
    pyMass createMass(){pyMass pm(sim.createMass());  return pm;}
//    pyMass createMass(py::array_t<double>);

    pySpring createSpring(){ pySpring ps(sim.createSpring());  return ps;}
//    pySpring createSpring();

    // Delete
//    void deleteMass(pyMass pm);
//    void deleteSpring(pySpring ps);
//    void deleteContainer(Container * c);
//
//    void get(pyMass pm);
//    void get(pySpring ps);
//    void get(Container *c);
//
//    void set(pyMass pm);
//    void set(pySpring ps);
//    void set(Container * c);

    void getAll(){ sim.getAll();};
    void setAll(){ sim.setAll();};

    // Constraints
//    void createPlane(const Vec & abc, double d );
//    void createBall(const Vec & center, double r );

    void clearConstraints(){sim.clearConstraints();};

    // Containers
    Container * createContainer() {sim.createContainer();};
//    Cube * createCube(const Vec & center, double side_length);
//    Lattice * createLattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    // Bulk modifications, only update CPU
    void setSpringConstant(double k) {sim.setSpringConstant(k);}
    void setMassValues(double m){sim.setMassValues(m);}
    void setDeltaT(double dt){sim.setDeltaT(dt);}

    void defaultRestLength(){sim.defaultRestLength();}

    // Control
    void start(double time = 1E20) {sim.start(time);}

    void stop(){sim.stop();}
    void stop(double time){sim.stop(time);}

    void pause(double t){sim.pause(t);}
    void resume(){sim.resume();}

    void wait(double t){sim.wait(t);}

    double time() { return sim.time();}
    double running() { return sim.running();}

    void printPositions(){sim.printPositions();}
    void printForces(){sim.printForces();}

//    pyMass getMassByIndex(int i);
//    pySpring getSpringByIndex(int i);
//    Container * getContainerByIndex(int i);
};
#endif //LOCH_PYSIMULATION_H
