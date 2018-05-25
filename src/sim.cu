//
// Created by Jacob Austin on 5/17/18.
//

#include "sim.h"

#define G 9.81;

Simulation::~Simulation() {
    for (Mass * m : masses)
        delete m;

    for (Spring * s : springs)
        delete s;

    for (Constraint * c : constraints)
        delete c;

    for (ContainerObject * o : objs)
        delete o;
}

Mass * Simulation::createMass() {
    Mass * m = new Mass();
    masses.push_back(m);
    return m;
}

Spring * Simulation::createSpring() {
    Spring * s = new Spring();
    springs.push_back(s);
    return s;
}

Spring * Simulation::createSpring(Mass * m1, Mass * m2, double k, double len) {
    Spring * s = new Spring(m1, m2, k, len);
    springs.push_back(s);
    return s;
}

void Simulation::setBreakpoint(double time) {
    bpts.insert(time);
}

CUDA_MASS * Simulation::massToArray() {
    CUDA_MASS * d_mass;
    cudaMalloc((void **)&d_mass, sizeof(CUDA_MASS) * masses.size());

    Mass * data = new Mass[masses.size()];

    Mass * h_iter = data;
    CUDA_MASS * d_iter = d_mass;

    for (Mass * m : masses) {
        CUDA_MASS temp(*m);
        memcpy(h_iter, &temp, sizeof(CUDA_MASS));
        m -> arrayptr = d_iter;
        h_iter++;
        d_iter++;
    }

    cudaMemcpy(d_mass, data, sizeof(CUDA_MASS) * masses.size());

    delete [] data;

    this -> d_mass = d_mass;

    return d_mass;
}

void Simulation::toArray() {
    CUDA_MASS * d_mass = massToArray();
    CUDA_SPRING * d_spring = springToArray();
}

void Simulation::fromArray() {
    massFromArray();

    delete [] d_spring;
    delete [] d_mass;
//    Spring * data = spring_arr;
//
//    for (Spring * s : springs) {
//        memcpy(s, data, sizeof(Spring));
//        s -> setMasses(masses[(data -> _left) - mass_arr], masses[(data -> _right) - mass_arr]);
//        data += sizeof(Spring);
//    }
}

void Simulation::massFromArray() {
    CUDA_MASS * h_data = new CUDA_MASS[masses.size()];
    cudaMemcpy(d_mass, h_data, sizeof(CUDA_MASS) * masses.size());

    for (int i = 0; i < masses.size(); i++) {
        *masses[i] = h_data[i];
    }

    delete [] h_data;

    cudaFree(d_mass);
}

CUDA_SPRING * Simulation::springToArray() {
    CUDA_SPRING * d_spring;
    cudaMalloc((void **)& d_spring, sizeof(CUDA_SPRING) * springs.size());

    Spring * h_spring = new Spring[springs.size()];

    Spring * h_iter = h_spring;
//    CUDA_SPRING * d_iter = d_spring;

    for (Spring * s : springs) {
        CUDA_SPRING temp(*s);
        temp._left = s -> _left -> arrayptr;
        temp._right = s -> _right -> arrayptr;
        memcpy(h_iter, &temp, sizeof(CUDA_SPRING));
        h_iter++;
    }

    cudaMemcpy(d_spring, h_spring, sizeof(CUDA_SPRING) * springs.size());

    delete [] h_spring;

    this -> d_spring = d_spring;

    return d_spring;
}

void Simulation::springFromArray() {
    cudaFree(d_spring);
}

__global__ void computeSpringForces(CUDA_SPRING * d_spring, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_springs ) {
        CUDA_SPRING & spring = d_spring[i];
        Vec temp = (spring._right -> getPosition()) - (spring._left -> getPosition());
        Vec force = spring._k * (spring._rest - temp.norm()) * (temp / temp.norm());
        spring.right -> force += force;
        spring.left -> force -= force;
    }
}

__global__ void computeMassForces(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.force += Vec(0, 0, - G * mass.m);
    }
}


__global__ void update(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.acc = mass.force / mass.m;
        mass.vel = mass.vel + mass.acc * mass.dt;
        mass.pos = mass.pos + mass.vel * mass.dt;
    }

    mass.time += mass.dt;
    mass.force = Vec(0, 0, 0);
}


void Simulation::resume() {
    int threadsPerBlock = 256;

    RUNNING = 1;
    toArray();

    while (1) {
        T += dt;

        if (!bpts.empty() && *bpts.begin() <= T) {
            bpts.erase(bpts.begin());
            fromArray();
            RUNNING = 0;
            break;
        }


        int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
        int springBlocksPerGrid = (springs.size() + threadsPerBlock - 1) / threadsPerBlock;

        computeSpringForces<<<springBlocksPerGrid, threadsPerBlock>>>(d_spring, springs.size()); // KERNEL
        computeMassForces<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, masses.size()); // KERNEL
        update<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, masses.size());
    }
}

int compareMass(const Mass * x, const Mass * y) { // Compare two masses' dts
    return x -> deltat() < y -> deltat() ? 0 : 1;
}

void Simulation::run() { // repeatedly run next
    T = 0;
    dt = 0.01; // (*std::min_element(masses.begin(), masses.end(), compareMass)) -> deltat();
    
    resume(); //Start the simulation for the first time
}

Plane * Simulation::createPlane(const Vec & abc, double d ) { // creates half-space ax + by + cz < d
    Plane * new_plane = new Plane(abc, d);
    constraints.push_back(new_plane);
    return new_plane;
}

Cube * Simulation::createCube(const Vec & center, double side_length) { // creates half-space ax + by + cz < d
    Cube * cube = new Cube(center, side_length);
    for (Mass * m : cube -> masses) {
        masses.push_back(m);
    }

    for (Spring * s : cube -> springs) {
        springs.push_back(s);
    }

    objs.push_back(cube);

    return cube;
}

void Simulation::printPositions() {
    for (Mass * m : masses) {
            std::cout << m->getPosition() << std::endl;
    }

    std::cout << std::endl;
}

void Simulation::printForces() {
    for (Mass * m : masses) {
        std::cout << m->getForce() << std::endl;
    }

    std::cout << std::endl;
}
