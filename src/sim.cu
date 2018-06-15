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

    cudaFree(d_spring);
    cudaFree(d_mass);

#ifdef GRAPHICS
    glDeleteBuffers(1, &vertices);
    glDeleteBuffers(1, &colors);
    glDeleteBuffers(1, &indices);

    glDeleteProgram(programID);
    glDeleteVertexArrays(1, &VertexArrayID);

    glfwTerminate();
#endif
}

Mass * Simulation::createMass() {
    Mass * m = new Mass();
    masses.push_back(m);
    return m;
}

Mass * Simulation::createMass(const Vec & pos) {
    Mass * m = new Mass(pos);
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

void Simulation::setSpringConstant(double k) {
    for (Spring * s : springs) {
        s -> setK(k);
    }
}

void Simulation::defaultRestLength() {
    for (Spring * s : springs) {
        s -> defaultLength();
    }
}

void Simulation::setMass(double m) {
    for (Mass * mass : masses) {
        mass -> setMass(m);
    }
}
void Simulation::setMassDeltaT(double dt) {
    for (Mass * m : masses) {
        m -> setDeltaT(dt);
    }
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

void Simulation::constraintsToArray() {
    d_constraints.reserve(constraints.size());

    for (Constraint * c : constraints) {
        Constraint * d_temp;
        cudaMalloc((void **)& d_temp, sizeof(Constraint));
        cudaMemcpy(d_temp, c, sizeof(Constraint), cudaMemcpyHostToDevice);
        d_constraints.push_back(d_temp);
    }
}

void Simulation::toArray() {
    CUDA_MASS * d_mass = massToArray(); // must come first
    CUDA_SPRING * d_spring = springToArray();
    constraintsToArray();
}

void Simulation::massFromArray() {
    CUDA_MASS * h_mass = new CUDA_MASS[masses.size()];
    cudaMemcpy(h_mass, d_mass, sizeof(CUDA_MASS) * masses.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < masses.size(); i++) {
        *masses[i] = Mass(h_mass[i]);
    }

    delete [] h_mass;

//    cudaFree(d_mass);
}

void Simulation::springFromArray() {
//    cudaFree(d_spring);
}

void Simulation::constraintsFromArray() {
    //
}

void Simulation::fromArray() {
    massFromArray();
    springFromArray();
    constraintsFromArray();
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

//__global__ void printSpringForce(CUDA_SPRING * d_springs, int num_springs) {
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (i < num_springs) {
//        printf("%d: left: (%5f, %5f, %5f), right:  (%5f, %5f, %5f)\n\n ", i, d_springs[i]._left -> force[0], d_springs[i]._left -> force[1], d_springs[i]._left -> force[2], d_springs[i]._right -> force[0], d_springs[i]._right -> force[1], d_springs[i]._right -> force[2]);
//    }
//}

__global__ void computeSpringForces(CUDA_SPRING * d_spring, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < num_springs ) {
        CUDA_SPRING & spring = d_spring[i];
        Vec temp = (spring._right -> pos) - (spring._left -> pos);
        Vec force = spring._k * (spring._rest - temp.norm()) * (temp / temp.norm());

        if (spring._right -> fixed == 0) {
            spring._right->force.atomicVecAdd(force); // need atomics here
        }
        if (spring._left -> fixed == 0) {
            spring._left->force.atomicVecAdd(-force);
        }
    }
}

__global__ void computeMassForces(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        if (mass.fixed == 0) {
            mass.force += Vec(0, 0, -9.81 * mass.m); // can use += since executed after springs

            if (mass.pos[2] < 0)
                mass.force += Vec(0, 0, -10000 * mass.pos[2]);
        }
    }
}


__global__ void update(CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS & mass = d_mass[i];
        if (mass.fixed == 0) {
            mass.acc = mass.force / mass.m;
            mass.vel = mass.vel + mass.acc * mass.dt;
            mass.pos = mass.pos + mass.vel * mass.dt;
        }
//        mass.T += mass.dt;
        mass.force = Vec(0, 0, 0);
    }
}

__global__ void massForcesAndUpdate(CUDA_MASS * d_mass, Constraint ** d_constraints, int num_masses, int num_constraints) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        CUDA_MASS &mass = d_mass[i];

        if (mass.fixed == 1)
            return;

        for (int j = 0; j < num_constraints; j++) {
            if (i == 0) {
                printf("(%f, %f, %f)\n", d_constraints[j] -> getForce(mass.pos)[0], d_constraints[j] -> getForce(mass.pos)[1], d_constraints[j] -> getForce(mass.pos)[2]);
            }

            mass.force += d_constraints[j] -> getForce(mass.pos);
        }

        mass.force += Vec(0, 0, -9.81 * mass.m); // don't need atomics

        if (i == 0) {
            printf("num constraints: %d\n", num_constraints);
        }

//        if (mass.pos[2] < 0)
//            mass.force += Vec(0, 0, -10000 * mass.pos[2]); // don't need atomics

        mass.acc = mass.force / mass.m;
        mass.vel = mass.vel + mass.acc * mass.dt;
        mass.pos = mass.pos + mass.vel * mass.dt;

        mass.force = Vec(0, 0, 0);
    }
}

#ifdef GRAPHICS
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
#endif

void Simulation::resume() {
    int threadsPerBlock = 256;

    RUNNING = 1;
    toArray();

    while (1) {
        if (!bpts.empty() && *bpts.begin() <= T) {
            cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions
            bpts.erase(bpts.begin());
            fromArray();
            RUNNING = 0;
            break;
        }

        int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
        int springBlocksPerGrid = (springs.size() + threadsPerBlock - 1) / threadsPerBlock;

        if (massBlocksPerGrid > MAX_BLOCKS) {
            massBlocksPerGrid = MAX_BLOCKS;
        }

        if (springBlocksPerGrid > MAX_BLOCKS) {
            springBlocksPerGrid = MAX_BLOCKS;
        }

        cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions

#ifdef GRAPHICS
        computeSpringForces<<<springBlocksPerGrid, threadsPerBlock>>>(d_spring, springs.size()); // compute mass forces after syncing
        massForcesAndUpdate<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, thrust::raw_pointer_cast(&d_constraints[0]), masses.size(), d_constraints.size());//        T += dt;
        T += dt;
#else
        for (int i = 0; i < NUM_QUEUED_KERNELS; i++) {
            computeSpringForces<<<springBlocksPerGrid, threadsPerBlock>>>(d_spring, springs.size()); // compute mass forces after syncing
            massForcesAndUpdate<<<massBlocksPerGrid, threadsPerBlock>>>(d_mass, thrust::raw_pointer_cast(&d_constraints[0]), masses.size(), d_constraints.size());//        T += dt;
            T += dt;
        }
#endif

#ifdef GRAPHICS
        if (fmod(T, 250 * dt) < dt) {
            clearScreen();

            updateBuffers();
            draw();

            for (Constraint * c : constraints) {
                c -> draw();
            }

            renderScreen();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE ) == GLFW_PRESS || glfwWindowShouldClose(window) != 0) {
                RUNNING = 0;
                break;
            }
        }
#endif

    }
}

void Simulation::run() { // repeatedly run next
    T = 0;
    dt = 1000000;
    for (Mass * m : masses) {
        if (m -> deltat() < dt)
            dt = m -> deltat();
    }

#ifdef GRAPHICS
    this -> window = createGLFWWindow();

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    this -> programID = LoadShaders("shaders/TransformVertexShader.vertexshader", "shaders/ColorFragmentShader.fragmentshader");
    // Get a handle for our "MVP" uniform
    this -> MatrixID = glGetUniformLocation(programID, "MVP");

    this -> MVP = getProjection();

    generateBuffers();

    for (Constraint * c : constraints) {
        c -> generateBuffers();
    }
#endif

    resume();
}

#ifdef GRAPHICS

void Simulation::generateBuffers() {
    {
        GLuint colorbuffer; // bind colors to buffer colorbuffer
        glGenBuffers(1, &colorbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, 3 * masses.size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject(colorbuffer);
        this -> colors = colorbuffer;
    }

    {
        GLuint elementbuffer; // create buffer for main cube object
        glGenBuffers(1, &elementbuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * springs.size() * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW); // second argument is number of bytes
        cudaGLRegisterBufferObject(elementbuffer);
        this -> indices = elementbuffer;
    }

    {
        GLuint vertexbuffer;
        glGenBuffers(1, &vertexbuffer); // bind cube vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, 3 * masses.size() * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject(vertexbuffer);
        this -> vertices = vertexbuffer;
    }
}

__global__ void updateVertices(float * gl_ptr, CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        gl_ptr[3 * i] = (float) d_mass[i].pos[0];
        gl_ptr[3 * i + 1] = (float) d_mass[i].pos[1];
        gl_ptr[3 * i + 2] = (float) d_mass[i].pos[2];
    }
}

__global__ void updateIndices(unsigned int * gl_ptr, CUDA_SPRING * d_spring, CUDA_MASS * d_mass, int num_springs) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_springs) {
        gl_ptr[2*i] = (d_spring[i]._left) - d_mass;
        gl_ptr[2*i + 1] = (d_spring[i]._right) - d_mass;
    }
}

__global__ void updateColors(float * gl_ptr, CUDA_MASS * d_mass, int num_masses) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_masses) {
        gl_ptr[3 * i] = (float) d_mass[i].color[0];
        gl_ptr[3 * i + 1] = (float) d_mass[i].color[1];
        gl_ptr[3 * i + 2] = (float) d_mass[i].color[2];
    }
}

void Simulation::updateBuffers() {
    int threadsPerBlock = 1024;

    int massBlocksPerGrid = (masses.size() + threadsPerBlock - 1) / threadsPerBlock;
    int springBlocksPerGrid = (springs.size() + threadsPerBlock - 1) / threadsPerBlock;

    if (update_colors) {
        glBindBuffer(GL_ARRAY_BUFFER, colors);
        void *colorPointer; // if no masses, springs, or colors are changed/deleted, this can be run only once
        cudaGLMapBufferObject(&colorPointer, colors);
        updateColors<<<massBlocksPerGrid, threadsPerBlock>>>((float *) colorPointer, d_mass, masses.size());
        cudaGLUnmapBufferObject(colors);
        update_colors = 0;
    }


    if (update_indices) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);
        void *indexPointer; // if no masses or springs are deleted, this can be run only once
        cudaGLMapBufferObject(&indexPointer, indices);
        updateIndices<<<springBlocksPerGrid, threadsPerBlock>>>((unsigned int *) indexPointer, d_spring, d_mass, springs.size());
        cudaGLUnmapBufferObject(indices);
        update_indices = 0;
    }

    {
        glBindBuffer(GL_ARRAY_BUFFER, vertices);
        void *vertexPointer;
        cudaGLMapBufferObject(&vertexPointer, vertices);
        updateVertices<<<massBlocksPerGrid, threadsPerBlock>>>((float *) vertexPointer, d_mass, masses.size());
        cudaGLUnmapBufferObject(vertices);
    }
}

void Simulation::draw() {
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, this -> vertices);
    glPointSize(this -> pointSize);
    glLineWidth(this -> lineWidth);
    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, this -> colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    glDrawArrays(GL_POINTS, 0, masses.size()); // 3 indices starting at 0 -> 1 triangle
    glDrawElements(GL_LINES, 2 * springs.size(), GL_UNSIGNED_INT, (void*) 0); // 3 indices starting at 0 -> 1 triangle

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

#endif

struct setup {
    template <typename Tuple>
    __host__ __device__ void operator()(const Tuple &arg) {
        (thrust::get<0>(arg)).set_values(thrust::get<1>(arg), thrust::get<2>(arg));
        (thrust::get<3>(arg)).set_values(&thrust::get<0>(arg));
    }
};


Plane * Simulation::createPlane(const Vec & abc, double d ) { // creates half-space ax + by + cz < d
    d_vecs.push_back(abc);
    d_doubles.push_back(d);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_planes.begin(), d_vecs.begin(), d_doubles.begin(), d_constraints.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_planes.end(), d_vecs.end(), d_doubles.end(), d_constraints.end())), setup());
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

    for (Spring * s : cube -> springs) {
        s -> setRestLength((s -> _right -> getPosition() - s -> _left -> getPosition()).norm());
    }

    return cube;
}

Lattice * Simulation::createLattice(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    Lattice * l = new Lattice(center, dims, nx, ny, nz);

    for (Mass * m : l -> masses) {
        masses.push_back(m);
    }

    for (Spring * s : l -> springs) {
        springs.push_back(s);
    }

    objs.push_back(l);

    for (Spring * s : l -> springs) {
        s -> setRestLength((s -> _right -> getPosition() - s -> _left -> getPosition()).norm());
    }

    return l;
}

Ball * Simulation::createBall(const Vec & center, double r ) { // creates ball with radius r at position center
    d_vecs.push_back(center);
    d_doubles.push_back(r);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_balls.begin(), d_vecs.begin(), d_doubles.begin(), d_constraints.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_balls.end(), d_vecs.end(), d_doubles.end(), d_constraints.end())), setup());
    Ball * new_ball = new Ball(center, r);
    constraints.push_back(new_ball);
    return new_ball;
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