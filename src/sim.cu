//
// Created by Jacob Austin on 5/17/18.
//

#include "sim.h"

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

    CUDA_MASS * data = new CUDA_MASS[masses.size()];

    CUDA_MASS * h_iter = data;
    CUDA_MASS * d_iter = d_mass;

    for (Mass * m : masses) {
        *h_iter = CUDA_MASS(*m);
        m -> arrayptr = d_iter;
        h_iter++;
        d_iter++;
    }

    cudaMemcpy(d_mass, data, sizeof(CUDA_MASS) * masses.size(), cudaMemcpyHostToDevice);

    delete [] data;

    this -> d_mass = d_mass;

    return d_mass;
}

CUDA_SPRING * Simulation::springToArray() {
    CUDA_SPRING * d_spring;
    cudaMalloc((void **)& d_spring, sizeof(CUDA_SPRING) * springs.size());

    CUDA_SPRING * h_spring = new CUDA_SPRING[springs.size()];

    CUDA_SPRING * h_iter = h_spring;
//    CUDA_SPRING * d_iter = d_spring;

    for (Spring * s : springs) {
        *h_iter = CUDA_SPRING(*s, s -> _left -> arrayptr, s -> _right -> arrayptr);
        h_iter++;
    }

    cudaMemcpy(d_spring, h_spring, sizeof(CUDA_SPRING) * springs.size(), cudaMemcpyHostToDevice);

    delete [] h_spring;

    this -> d_spring = d_spring;

    return d_spring;
}

void Simulation::toArray() {
    CUDA_MASS * d_mass = massToArray();
    CUDA_SPRING * d_spring = springToArray();
}

void Simulation::massFromArray() {
    CUDA_MASS * h_data = new CUDA_MASS[masses.size()];
    cudaMemcpy(h_data, d_mass, sizeof(CUDA_MASS) * masses.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < masses.size(); i++) {
        *masses[i] = Mass(h_data[i]);
    }

    delete [] h_data;

    cudaFree(d_mass);
}

void Simulation::springFromArray() {
    cudaFree(d_spring);
}


void Simulation::fromArray() {
    massFromArray();
    springFromArray();
}

__global__ void printMasses(CUDA_MASS * d_masses, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d", 5);

    if (i < 50) {
        printf("hello");
        //d_masses[i].pos.print();
    }
}

__global__ void computeSpringForces(CUDA_SPRING * d_spring, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_springs ) {
        CUDA_SPRING & spring = d_spring[i];
        Vec temp = (spring._right -> pos) - (spring._left -> pos);
        Vec force = spring._k * (spring._rest - temp.norm()) * (temp / temp.norm());
        spring._right -> force += force;
        spring._left -> force += -force;
    }
}

__global__ void computeMassForces(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.force += Vec(0, 0, - 9.81 * mass.m);
    }
}


__global__ void update(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.acc = mass.force / mass.m;
        mass.vel = mass.vel + mass.acc * mass.dt;
        mass.pos = mass.pos + mass.vel * mass.dt;

        mass.T += mass.dt;
        mass.force = Vec(0, 0, 0);
    }
}

void Simulation::resume() {
    int threadsPerBlock = 256;

    RUNNING = 1;
    toArray();

    printPositions();

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
    if (RUNNING) {
        std::cout << "\nDEVICE MASSES: " << std::endl;
        int threadsPerBlock = 256;
        int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << massBlocksPerGrid << " " << threadsPerBlock << std::endl;
        printMasses<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, masses.size());
        cudaDeviceSynchronize();
    }
    else {
        std::cout << "\nHOST MASSES: " << std::endl;
        for (Mass * m : masses) {
            std::cout << m->getPosition() << std::endl;
        }
    }

    std::cout << std::endl;
}

//void Simulation::printForces() {
//    for (Mass * m : masses) {
//        std::cout << m->getForce() << std::endl;
//    }
//
//    std::cout << std::endl;
//}
