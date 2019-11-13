# Titan
**Titan** is a versitile CUDA-based physics simulation library that provides a GPU-accelerated environment for physics primatives like springs and masses. Library users can create masses, springs, and more complicated objects, apply constraints, and modify simulation parameters in real time, while the simulation runs asynchronously on the GPU. Titan has been used for GPU-accelerated reinforcement learning, physics and biomedical research, and topology optimization.

## Installation

Detailed instructions can be found in the [user wiki](https://github.com/ja3067/Titan/wiki/Set-Up) for building and installing the Titan physics library.

**Try a simple Titan physics simulation**

```C++
#include <Titan/sim.h>

int main() {
  Simulation sim;
  sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 5, 5, 5); // create lattice with center at (0, 0, 10) and given dimensions
  sim.createPlane(Vec(0, 0, 1), 0); // create constraint plane
  sim.start();
}
```

This simple program produces a large lattice bouncing on the given plane:

<img src="https://i.imgur.com/zdB0ZPg.gif" width="400" height="400">

For more examples and troubleshooting, see the [github wiki](https://github.com/ja3067/Titan/wiki/Set-Up). 

## About

This software was written by Jacob Austin and Rafael Corrales Fatou as part of a project led by Professor Hod Lipson at the Creative Machines Lab at Columbia University. This software is released under an Apache 2.0 license.
