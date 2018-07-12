# Loch
A CUDA-based physics simulation sandbox written in C++ which uses springs and masses to simulate flexible robots and other mechanical objects. 

## About The Library

This library exposes a simple and powerful C++ API for handling general spring/mass simulations on an NVIDIA GPU. A user can create an arbitrary configuration of masses and springs with custom constraints, and run a simulation asynchronously while performing computations on the CPU and updating parameters in real time. A minimal example involves as little setup as:

```C++
int main() {
  Simulation sim;
  Mass * m1 = sim.createMass(Vec(0, 0, 1));
  Mass * m2 = sim.createMass(Vec(0, 0, -1));
  Spring * s1 = sim.createMass(m1, m2);
  
  sim.start();
}
```

The simulation is dynamically parallelized on the GPU and runs orders of magnitude faster than any available CPU implementation. Benchmarks have shown the library running as fast as 400 million springs per second.

## Installation Instructions

Detailed instructions can be found in the user wiki [here](https://github.com/ja3067/Loch/wiki/Set-Up).
