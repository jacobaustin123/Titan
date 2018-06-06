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

    glDeleteProgram(programID);
    glDeleteVertexArrays(1, &VertexArrayID);

    glfwTerminate();
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

//CUDA_MASS * Simulation::ToArray() {
//    CUDA_MASS * d_mass;
//    cudaMalloc((void **)&d_mass, sizeof(CUDA_MASS) * masses.size());
//
//    CUDA_MASS * data = new CUDA_MASS[masses.size()];
//
//    CUDA_MASS * h_iter = data;
//    CUDA_MASS * d_iter = d_mass;
//
//    for (Mass * m : masses) {
//        *h_iter = CUDA_MASS(*m);
//        m -> arrayptr = d_iter;
//        h_iter++;
//        d_iter++;
//    }
//
//    cudaMemcpy(d_mass, data, sizeof(CUDA_MASS) * masses.size(), cudaMemcpyHostToDevice);
//
//    delete [] data;
//
//    this -> d_mass = d_mass;
//
//    return d_mass;

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
    CUDA_MASS * d_mass = massToArray(); // must come first
    CUDA_SPRING * d_spring = springToArray();
}

void Simulation::massFromArray() {
    CUDA_MASS * h_mass = new CUDA_MASS[masses.size()];
    cudaMemcpy(h_mass, d_mass, sizeof(CUDA_MASS) * masses.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < masses.size(); i++) {
        *masses[i] = Mass(h_mass[i]);
    }

    delete [] h_mass;

    cudaFree(d_mass);
}

void Simulation::springFromArray() {

    CUDA_SPRING *h_spring = new CUDA_SPRING[springs.size()];
    cudaMemcpy(h_spring, d_spring, sizeof(CUDA_SPRING) * springs.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < springs.size(); i++) {
        *springs[i] = Spring(h_spring[i]);
    }

    delete [] h_spring;

    cudaFree(d_spring);
}


void Simulation::fromArray() {
    massFromArray();
    springFromArray();
}

__global__ void printMasses(CUDA_MASS * d_masses, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        d_masses[i].pos.print();
    }
}

__global__ void printForce(CUDA_MASS * d_masses, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        d_masses[i].force.print();
    }
}

__global__ void printSpring(CUDA_SPRING * d_springs, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_springs) {
        printf("%d: left: (%5f, %5f, %5f), right:  (%5f, %5f, %5f)\n\n ", i, d_springs[i]._left -> pos[0], d_springs[i]._left -> pos[1], d_springs[i]._left -> pos[2], d_springs[i]._right -> pos[0], d_springs[i]._right -> pos[1], d_springs[i]._right -> pos[2]);
    }
}

__global__ void printSpringForce(CUDA_SPRING * d_springs, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_springs) {
        printf("%d: left: (%5f, %5f, %5f), right:  (%5f, %5f, %5f)\n\n ", i, d_springs[i]._left -> force[0], d_springs[i]._left -> force[1], d_springs[i]._left -> force[2], d_springs[i]._right -> force[0], d_springs[i]._right -> force[1], d_springs[i]._right -> force[2]);
    }
}

__global__ void computeSpringForces(CUDA_SPRING * d_spring, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_springs ) {
        CUDA_SPRING & spring = d_spring[i];
        Vec temp = (spring._right -> pos) - (spring._left -> pos);
        Vec force = spring._k * (spring._rest - temp.norm()) * (temp / temp.norm());

//        if (i == 0)
//            printf("(%5f, %5f, %5f)\n", force[0], force[1], force[2]);

        spring._right -> force.atomicVecAdd(force);
        spring._left -> force.atomicVecAdd(-force);
    }
}

__global__ void computeMassForces(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.force.atomicVecAdd(Vec(0, 0, - 9.81 * mass.m));

        if (mass.pos[2] < 0)
            mass.force.atomicVecAdd(Vec(0, 0, - 10000 * mass.pos[2]));
    }
}


__global__ void update(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        mass.acc = mass.force / mass.m;
        mass.vel = mass.vel + mass.acc * mass.dt;
        mass.pos = mass.pos + mass.vel * mass.dt;

//        mass.T += mass.dt;
        mass.force = Vec(0, 0, 0);
    }
}

void Simulation::clearScreen() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

    // Use our shader
    glUseProgram(programID);

    // Send our transformation to the currently bound shader in the "MVP" uniform
    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
}

void Simulation::renderScreen() {
    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Simulation::resume() {
    int threadsPerBlock = 1024;

    RUNNING = 1;
    toArray();

//    printSprings();
//    printSpringForces();
//
//    printPositions();

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
        cudaDeviceSynchronize(); /*Necessary? Why halt CPU until end of compute Sprinf and MAss if we nee to call
        another kernel afterwards*/

        update<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, masses.size());

        if (fmod(T, 250 * dt) < dt) {
            fromArray();

            clearScreen();

            for (ContainerObject * c : objs) {
                c -> updateBuffers();
                c -> draw();
            }

            for (Constraint * c : constraints) {
                c -> draw();
            }

            renderScreen();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS || glfwWindowShouldClose(window) != 0) {
                RUNNING = 0;
                break;
            }

            toArray();
        }
    }
}

int compareMass(const Mass * x, const Mass * y) { // Compare two masses' dts
    return x -> deltat() < y -> deltat() ? 0 : 1;
}

void Simulation::run() { // repeatedly run next
    T = 0;
    dt = 1000000;
    for (Mass * m : masses) {
        if (m -> deltat() < dt)
            dt = m -> deltat();
    }

//    dt = (*std::min_element(masses.begin(), masses.end(), cmp)) -> deltat();

    this -> window = createGLFWWindow();

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    this -> programID = LoadShaders("shaders/TransformVertexShader.vertexshader", "shaders/ColorFragmentShader.fragmentshader");
    // Get a handle for our "MVP" uniform
    this -> MatrixID = glGetUniformLocation(programID, "MVP");

    this -> MVP = getProjection();

    for (ContainerObject * c : objs) {
        c -> generateBuffers();
    }

    for (Constraint * c : constraints) {
        c -> generateBuffers();
    }

    resume();
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
        int threadsPerBlock = 1024;
        int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
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

void Simulation::printSprings() {
    if (RUNNING) {
        std::cout << "\nDEVICE SPRINGS: " << std::endl;
        int threadsPerBlock = 1024;
        int springBlocksPerGrid = (springs.size() + threadsPerBlock - 1) / threadsPerBlock;
        printSpring<<<springBlocksPerGrid, threadsPerBlock>>>(d_spring, springs.size());
        cudaDeviceSynchronize();
    }
    else {
        std::cout << "\nHOST SPRINGS: " << std::endl;
        for (Spring * s : springs) {
            std::cout << s->_left->getPosition() << s->_right->getPosition() << std::endl;
        }
    }

    std::cout << std::endl;
}

void Simulation::printSpringForces() {
    if (RUNNING) {
        std::cout << "\nDEVICE SPRINGS: " << std::endl;
        int threadsPerBlock = 1024;
        int springBlocksPerGrid = (springs.size() + threadsPerBlock - 1) / threadsPerBlock;
        printSpringForce<<<springBlocksPerGrid, threadsPerBlock>>>(d_spring, springs.size());
        cudaDeviceSynchronize();
    }
    else {
        std::cout << "\nHOST SPRINGS: " << std::endl;
        for (Spring * s : springs) {
            std::cout << s->_left->getForce() << s->_right->getForce() << std::endl;
        }
    }

    std::cout << std::endl;
}

void Simulation::printForces() {
    if (RUNNING) {
        std::cout << "\nDEVICE FORCES: " << std::endl;
        int threadsPerBlock = 1024;
        int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
        printForce<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, masses.size());
        cudaDeviceSynchronize();
    }
    else {
        std::cout << "\nHOST FORCES: " << std::endl;
        for (Mass * m : masses) {
            std::cout << m->getForce() << std::endl;
        }
    }

    std::cout << std::endl;
}