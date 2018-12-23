

#ifndef LOCH_PYOBJECT_H
#define LOCH_PYOBJECT_H


#include "object.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class pyContainer { // user mass class used to store real mass class properties in python and avoid ownership issues
public:
    //constructor
    pyContainer() = default;
    pyContainer(Container * containerp){ pointer = containerp;}

    //Pointer to real C++ mass object
    Container * pointer;
    //Properties
    double m() { return (pointer -> m);}  // mass in kg
    void m(double m) {pointer -> m = m;} // set mass
    double dt() { return (pointer -> dt);} // update interval
    double T() { return (pointer -> T);} // local time
    py::array_t<double> pos(); // position in m
    void pos(py::array_t<double> arr); // set position in m
    py::array_t<double> vel();
    py::array_t<double> acc();
    py::array_t<double> force();

};

class Container { // contains and manipulates groups of masses and springs
public:
    virtual ~Container() {};
    void translate(const Vec & displ); // translate all masses by fixed amount
    void rotate(const Vec & axis, double angle); // rotate all masses around a fixed axis by a specified angle with respect to the center of mass.

    void setMassValues(double m); // set masses for all Mass objects
    void setSpringConstants(double k); // set k for all Spring objects
    void setDeltaT(double dt); // set delta-t of all masses to dt
    void setRestLengths(double len); // set masses for all Mass objects

#ifdef CONSTRAINTS
    void addConstraint(CONSTRAINT_TYPE type, const Vec & v, double d);
    void clearConstraints();
#endif

    void makeFixed();

    void add(Mass * m);
    void add(Spring * s);
    void add(Container * c);

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Cube : public Container {
public:
    ~Cube() {};

    Cube(const Vec & center, double side_length = 1.0);

    double _side_length;
    Vec _center;
};

class Lattice : public Container {
public:
    ~Lattice() {};

    Lattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    int nx, ny, nz;
    Vec _center, _dims;
};

class Beam : public Container {
public:
    ~Beam() {};

    Beam(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    int nx, ny, nz;
    Vec _center, _dims;
};

#endif //LOCH_PYOBJECT_H
