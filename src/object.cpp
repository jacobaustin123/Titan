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
    return (disp < 0) ? disp * position : 0 * position;
}