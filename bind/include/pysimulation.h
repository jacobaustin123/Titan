#ifndef LOCH_PYSIMULATION_H
#define LOCH_PYSIMULATION_H

#include "sim.h"
#include "pymass.h"
#include "pyspring.h"
#include "pyobject.h"

class pySimulation{
public:
    //constructor
    pySimulation() = default;
    Simulation sim;

    //Create
    pyMass createMass(){pyMass pm(sim.createMass());  return pm;}
    pyMass createMass(py::array_t<double> arr);

    pySpring createSpring(){ pySpring ps(sim.createSpring());  return ps;}
    pySpring createSpring(pyMass m1, pyMass m2 ){ pySpring ps (sim.createSpring(m1.pointer, m2.pointer)); return ps;}

    pyContainer createContainer() {pyContainer pc (sim.createContainer()); return pc;}

    //delete
    void deleteMass(pyMass pm);
    void deleteSpring(pySpring ps);
    void deleteContainer(pyContainer pc);

    //getters
    void get(pyMass pm) {sim.get(pm.pointer);}
    void get(pySpring ps){sim.get(ps.pointer);}
    void get(pyContainer pc){sim.get(pc.pointer);}
    void getMassByIndex(int i){sim.getMassByIndex(i);}
    void getSpringByIndex(int i){getSpringByIndex(i);}
    void getAll(){ sim.getAll();};

    //setters
    void set(pyMass pm);
    void set(pySpring ps);
    void set(pyContainer pc);
    void setAll(){ sim.setAll();};

    //Containers
    void createCube(py::array_t<double> center, double side_lenght);
    void createLattice(py::array_t<double> center, double side_lenght);
    void createRobot(py::array_t<double> center, double side_lenght);
    void createBeam(py::array_t<double> center, double side_lenght);
    void importFromSTL(std::string abc, double density );

    // Constraints
    void createPlane(py::array_t<double> abc, double d );
    void createBall(py::array_t<double> abc, double r );

    void clearConstraints(){sim.clearConstraints();};


    // Bulk modifications, only update CPU
    void setAllSpringConstantValues(double k) {sim.setAllSpringConstantValues(k);}
    void setAllMassValues(double m){ sim.setAllMassValues(m);}
    void setAllDeltaTValues(double dt){sim.setAllDeltaTValues(dt);}

    void defaultRestLength(){sim.defaultRestLength();}

    // Control
    void start() {sim.start();}

    void stop(){sim.stop();}
    void stop(double time){sim.stop(time);}

    void pause(double t){sim.pause(t);}
    void resume(){sim.resume();}
    void setBreakpoint(double time){sim.setBreakpoint(time);}

    void wait(double t){sim.wait(t);}
    void waitUntil(double t);
    void waitForEvent();

    double time() { return sim.time();}
    double running() { return sim.running();}

    void printPositions(){sim.printPositions();}
    void printForces(){sim.printForces();}

};
#endif //LOCH_PYSIMULATION_H
