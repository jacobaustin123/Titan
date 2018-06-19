//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_MASS_H
#define LOCH_MASS_H

#include "vec.h"

class Mass {
public:
    // constructors

    Mass() { m = 1.0; fixed = 0; delta_t = 0.01; t = 0; color = Vec(1.0, 0.2, 0.2); arrayptr = nullptr; }
    Mass(const Vec & position, double mass = 1.0, int fixed = 0, double dt = 0.0001) :
            m(mass), pos(position), fixed(fixed), delta_t(dt), t(0), color(Vec(1.0, 0.2, 0.2)), arrayptr(nullptr) {}; // defaults everything

    // setters
    void setMass(double m) { this -> m = m; };
    void setPos(const Vec & pos) { this -> pos = pos; }
    void setVel(const Vec & vel) { this -> vel = vel; }
    void setAcc(const Vec & acc) { this -> acc = acc; }
    void setForce(const Vec & force) { this -> force = force; }
    void setDeltaT(double dt) { this -> delta_t = dt; }
    void setColor(const Vec & color) { this -> color = color; }

    void stepTime() { t += delta_t; }

    // manipulate
    void translate(const Vec & displ) { this -> pos += displ; }
    void makeFixed() { fixed = 1; }
    void makeMovable() { fixed = 0; }

    // getters
    double getMass() { return m; }
    const Vec & getPosition() { return pos; }
    const Vec & getVelocity() { return vel; }
    const Vec & getAcceleration() { return acc; }
    const Vec & getForce() { return force; }
    double time() { return t; }
    double dt() const { return delta_t; }

    int isFixed() { return (fixed == 1); }

    // private methods
    void update(); // update pos, vel, and acc based on force
    void addForce(const Vec &); // add force vector to current force
    void resetForce(); // set force = 0;

    Mass * arrayptr;

    double m; // mass in kg
    double delta_t; // update interval
    double t; // local time
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2
    Vec color; // color of mass

    int fixed; // is the mass position fixed?
};

#endif //LOCH_MASS_H