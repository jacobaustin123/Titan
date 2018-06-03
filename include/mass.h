//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_MASS_H
#define LOCH_MASS_H

#include "vec.h"

class Mass {
public:
    Mass() { m = 1.0; fixed = 0; dt = 0.01; T = 0; }

    Mass(double mass, const Vec & position, int fixed = 0, double dt = 0.01) :
            m(mass), pos(position), fixed(fixed), dt(dt), T(0) {}; // defaults everything

    void setMass(double m) { this -> m = m; };
    void setPos(const Vec & pos) { this -> pos = pos; }
    void setVel(const Vec & vel) { this -> vel = vel; }
    void setAcc(const Vec & acc) { this -> acc = acc; }
    void setForce(const Vec & force) { this -> force = force; }
    void setDeltaT(double dt) { this -> dt = dt; }
    void translate(const Vec & displ) { this -> pos += displ; }

    void makeFixed() { fixed = 1; }
    void makeMovable() { fixed = 0; }

    int isFixed() { return (fixed == 1); }
    double getMass() { return m; }
    const Vec & getPosition() { return pos; }
    const Vec & getVelocity() { return vel; }
    const Vec & getAcceleration() { return acc; }
    const Vec & getForce() { return force; }
    double time() { return T; }
    double deltat() const { return dt; }
    void stepTime() { T += dt; }

    void update(); // update pos, vel, and acc based on force
    void addForce(const Vec &); // add force vector to current force
    void resetForce(); // set force = 0;

    Mass * arrayptr;
private:
    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2

    int fixed; // is the mass position fixed?
};

#endif //LOCH_MASS_H
