#ifndef LOCH_OBJECT_H
#define LOCH_OBJECT_H

#include <vector>
#include "mass.h"
#include "spring.h"
#include "vec.h"

static double DISPL_CONST = 10000;

// base class for larger objects like Cubes, etc.

class BaseObject { // base commands for all objects
public:
    virtual ~BaseObject() {};
    virtual void translate(const Vec & displ) = 0; // translate all masses by fixed amount
};

class Constraint : public BaseObject { // constraint like plane or sphere which applies force to masses
public:
    virtual Vec getForce(const Vec & position) = 0; // returns force on an object based on its position, e.g. plane or
};

class ContainerObject : public BaseObject { // contains and manipulates groups of masses and springs
public:
    void setMassValue(double m); // set masses for all Mass objects
    void setKValue(double k); // set k for all Spring objects
    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Ball : public Constraint { // ball constraint, force is inversely proportional to distance
public:
    void setRadius(double r) { _radius = r; }
    void setCenter(const Vec & center) { _center = center; }

private:
    double _radius;
    Vec _center;
};

class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
public:
    Plane(const Vec & normal, double d);
    Vec getForce(const Vec & position);
    void setNormal(const Vec & normal) { _normal = normal; }; // normal is (a, b, c)
    void setOffset(double d) { _offset = d; }; // ax + by + cz < d
    Vec _normal;
    double _offset;
    void translate(const Vec & displ);
};

class Cube : public ContainerObject {
public:
    Cube(const Vec & center, double side_length = 1.0);
    virtual ~Cube() {};

    void translate(const Vec & displ);

    double _side_length;
    Vec _center;
};

#endif //LOCH_OBJECT_H
