//
// Created by Jacob Austin on 5/21/18.
//

#include "object.h"

//class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
//public:
//    void setNormal(const Vec & normal) { _normal = normal; }; // normal is (a, b, c)
//    void setOffset(double d) { _offset = d; }; // ax + by + cz < d
//
//private:
//    Vec _normal;
//    double _offset;
//};

Vec Plane::getForce(const Vec & position) { // returns force on an object based on its position, e.g. plane or
    double disp = dot(position, _normal) - _offset;
    return (disp < 0) ? - DISPL_CONST * disp * _normal : 0 * _normal;
}

Plane::Plane(const Vec & normal, double d) {
    _offset = d;
    _normal = normal;
}

void Plane::translate(const Vec & displ) {
    _offset += dot(displ, _normal);
}

void ContainerObject::setMassValue(double m) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> setMass(m);
    }
}

void ContainerObject::setKValue(double k) {
    for (Spring * spring : springs) {
        spring -> setK(k);
    }
}

Cube::Cube(const Vec & center, double side_length) {
    _center = center;
    _side_length = side_length;

    for (int i = 0; i < 8; i++) {
        masses.push_back(new Mass(1.0, side_length * Vec(i & 1, (i >> 1) & 1, (i >> 2) & 1)));
    }

    for (int i = 0; i < 8; i++) { // add the appropriate springs
        for (int j = i + 1; j < 8; j++) {
            springs.push_back(new Spring(masses[i], masses[j]));
        }
    }
}

void Cube::translate(const Vec & displ) {
    for (Mass * m : masses) {
        m->translate(displ);
    }
}