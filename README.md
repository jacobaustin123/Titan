# Loch
**Loch** is a versitile CUDA-based physics simulation library exposing powerful GPU-accelerated operations on physics primatives like springs and masses. Library users can create masses, springs, and more complicated objects, apply constraints, and modify simulation parameters in real time, while the simulation runs asynchronous on the GPU.

## Installation

Detailed instructions can be found in the user wiki [here](https://github.com/ja3067/Loch/wiki/Set-Up) for downloading and building the Loch physics library.

** Try a simple Loch physics simulation **

```C++
#include <Loch/sim.h>

int main() {
  sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 5, 5, 5); // create lattice with center at (0, 0, 10) and given dimensions
  sim.createPlane(Vec(0, 0, 1), 0); // create constraint plane
  sim.start(10); // run for 10 seconds;
}
```
