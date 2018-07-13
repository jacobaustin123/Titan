# Loch
**Loch** is a versitile CUDA-based physics simulation library exposing powerful GPU-accelerated operations on physics primatives like springs and masses. Library users can create masses, springs, and more complicated objects, apply constraints, and modify simulation parameters in real time, while the simulation runs asynchronous on the GPU.

## Installation

Detailed instructions can be found in the [user wiki](https://github.com/ja3067/Loch/wiki/Set-Up) for building and installing the Loch physics library.

**Try a simple Loch physics simulation**

```C++
#include <Loch/sim.h>

int main() {
  sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 5, 5, 5); // create lattice with center at (0, 0, 10) and given dimensions
  sim.createPlane(Vec(0, 0, 1), 0); // create constraint plane
  sim.start(10); // run for 10 seconds;
}
```

This simple program produces a large lattice bouncing on the given plane:

<img src="https://i.imgur.com/zdB0ZPg.gif" width="400" height="400">

For more examples and troubleshooting, see the [github wiki](https://github.com/ja3067/Loch/wiki/Using-CMake-or-Visual-Studio). 

## License

This software was written by Jacob Austin and Rafael Corrales Fatou as part of a project led by Professor Hod Lipson at Columbia University. The software is currently closed-source, but may be open-sourced in the future. Please do not redistribute until that time.
