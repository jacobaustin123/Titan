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

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
const int ACTIVE_CONTRACT_THEN_EXPAND = 0;
const int ACTIVE_EXPAND_THEN_CONTRACT = 1;
const int PASSIVE_SOFT = 2;
const int PASSIVE_STIFF = 3;


class Spring {
public:

    //Properties
    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)

    // BREATHING
    int _type; // 0-3
    double _omega; // frequency
    
    Mass * _left; // pointer to left mass object // private
    Mass * _right; // pointer to right mass object

    //Set
    Spring() { _left = nullptr; _right = nullptr; arrayptr = nullptr; _k = 10000.0; _rest = 1.0; _type=PASSIVE_STIFF; _omega=0.0; }; //Constructor
    
    Spring(const CUDA_SPRING & spr);

    Spring(Mass * left, Mass * right, double k = 10000.0, double rest_len = 1.0) :
            _k(k), _rest(rest_len), _left(left), _right(right), arrayptr(nullptr) {}; //

    Spring(double k, double rest_length, Mass * left, Mass * right) :
            _k(k), _rest(rest_length), _left(left), _right(right) {};

    Spring(double k, double rest_length, Mass * left, Mass * right, int type, double omega) :
            _k(k), _rest(rest_length), _left(left), _right(right), _type(type), _omega(omega) {};
	    
    void setForce(); // will be private
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength(); //sets rest length

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; } //sets both right and left masses

    //Get
    Vec getForce(); // computes force on right object. left force is - right force.

private:
    //    Mass * _left; // pointer to left mass object // private
    //    Mass * _right; // pointer to right mass object
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
  int _type;
  double _omega;
};

#endif //TITAN_SPRING_H
