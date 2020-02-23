//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_SPRING_H
#define TITAN_SPRING_H

#include "mass.h"
#include "vec.h"

class Mass;
struct CUDA_SPRING;
struct CUDA_MASS;

enum SpringType {PASSIVE_SOFT, PASSIVE_STIFF, ACTIVE_CONTRACT_THEN_EXPAND, ACTIVE_EXPAND_THEN_CONTRACT};

class Spring {
public:
    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)

    SpringType _type; // 0-3, for oscillating springs
    double _omega; // frequency of oscillation
    double _damping; // damping on the masses.

    Mass * _left; // pointer to left mass object // private
    Mass * _right; // pointer to right mass object

    Spring() { 
        _left = nullptr; 
        _right = nullptr; 
        arrayptr = nullptr; 
        _k = 10000.0; 
        _rest = 1.0; 
        _type = PASSIVE_SOFT; 
        _omega = 0.0; 
        _damping = 0.0;
    };
    
    // Spring(const CUDA_SPRING & spr);

    Spring(Mass * left, Mass * right) {
        this -> _left = left;
        this -> _right = right;
        this -> defaultLength();
        this -> _k = 10000.0;
        this -> arrayptr = nullptr;
        _type = PASSIVE_SOFT;
        _omega = 0.0; 
        _damping = 0.0;
    };

    Spring(Mass * left, Mass * right, double k, double rest_length) {
        this -> _left = left;
        this -> _right = right;
        this -> _rest = rest_length;
        this -> _k = k;
        _type = PASSIVE_SOFT;
        _omega = 0.0; 
        _damping = 0.0;
    }

    void update(const CUDA_SPRING & spr);

    Spring(Mass * left, Mass * right, double k, double rest_length, SpringType type, double omega) :
            _k(k), _rest(rest_length), _left(left), _right(right), _type(type), _omega(omega) {};
	    
    void setForce(); // will be private
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength(); //sets rest length
    void changeType(SpringType type, double omega) { _type = type; _omega = omega;}
    void addDamping(double constant) { _damping = constant; }

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; } //sets both right and left masses

    Vec getForce(); // computes force on right object. left force is - right force.

private:
    CUDA_SPRING *arrayptr; //Pointer to struct version for GPU cudaMalloc

    friend class Simulation;
    friend struct CUDA_SPRING;
    friend class Container;
    friend class Lattice;
    friend class Cube;
    friend class Beam;
};

struct CUDA_SPRING {
  CUDA_SPRING() {};
  CUDA_SPRING(const Spring & s);
  
  CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right);
  
  CUDA_MASS * _left; // pointer to left mass object
  CUDA_MASS * _right; // pointer to right mass object
  
  double _k; // spring constant (N/m)
  double _rest; // spring rest length (meters)

  // Breathing
  SpringType _type;
  double _omega;
  double _damping;
};

#endif //TITAN_SPRING_H
