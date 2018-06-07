//
// Created by Jacob Austin on 5/17/18.
//

#include "sim.h"
#include <cmath>

Simulation::~Simulation() {
    for (Mass * m : masses)
        delete m;

    for (Spring * s : springs)
        delete s;

    for (Constraint * c : constraints)
        delete c;

#ifdef GRAPHICS
    glDeleteBuffers(1, &vertices);
    glDeleteBuffers(1, &colors);
    glDeleteBuffers(1, &indices);
    glDeleteProgram(programID);
    glDeleteVertexArrays(1, &VertexArrayID);

//     Close OpenGL window and terminate GLFW
    glfwTerminate();
#endif
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
    bpts.insert(Event(nullptr, time));
}

void Simulation::setSpringConstant(double k) {
    for (Spring * s : springs) {
        s -> setK(k);
    }
}

void Simulation::defaultRestLength() {
    for (Spring * s : springs) {
        s -> setRestLength((s ->_left->getPosition() - s -> _right->getPosition()).norm());
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

void Simulation::computeForces() {
    Spring * s = spring_arr;

    for (int i = 0; i < springs.size(); i++) { // update the forces
        s -> setForce();
        s++;
    }

    Mass * m = mass_arr; // constraints and gravity
    for (int i = 0; i < masses.size(); i++) {
        for (Constraint * c : constraints) {
            m -> addForce( c -> getForce(m -> getPosition()) ); // add force based on position relative to constraint
        }

        m -> addForce(Vec(0, 0, - m -> getMass() * G)); // add gravity

        m++;
    }
}

Mass * Simulation::massToArray() {
    Mass * data = new Mass[masses.size()];
    Mass * iter = data;

    for (Mass * m : masses) {
        memcpy(iter, m, sizeof(Mass));
        m -> arrayptr = iter;
        iter++;
    }

    this->mass_arr = data;

    return data;
}

Spring * Simulation::springToArray() {
    Spring * spring_data = new Spring[springs.size()];

    Spring * spring_iter = spring_data;

    for (Spring * s : springs) {
        memcpy(spring_iter, s, sizeof(Spring));
        spring_iter -> setMasses(s -> _left -> arrayptr, s -> _right -> arrayptr);
        spring_iter++;
    }

    this -> spring_arr = spring_data;

    return spring_data;
}

void Simulation::toArray() {
    Mass * mass_data = massToArray();
    Spring * spring_data = springToArray();
}

void Simulation::fromArray() {
    massFromArray();
    springFromArray();
}

void Simulation::massFromArray() {
    Mass * data = mass_arr;

    for (Mass * m : masses) {
        memcpy(m, data, sizeof(Mass));
        data ++;
    }

    delete[] mass_arr;
}

void Simulation::springFromArray() {
    delete [] spring_arr;
}

void Simulation::resume() {
    RUNNING = 1;
    toArray();

    while (1) {
        T += dt;

        if (!bpts.empty() && (*bpts.begin()).time <= T) {
            fromArray();

            if ((*bpts.begin()).func != nullptr) {
                (*bpts.begin()).func();
                if ((*bpts.begin()).repeat != 0) {
                    Event new_event = (*bpts.begin());
                    new_event.time += new_event.repeat;
                    bpts.insert(new_event);
                }
                bpts.erase(bpts.begin());
                resume();
                RUNNING = 0;
                break;
            } else {
                bpts.erase(bpts.begin());
                RUNNING = 0;
                break;
            }
        }


        computeForces(); // compute forces on all masses

        Mass * m = mass_arr;
        for (int i = 0; i < masses.size(); i++) {
            if (m -> time() <= T) { // !m -> isFixed()
                m -> stepTime();
                m -> update();
            }

            m -> resetForce();

            m++;
        }

#ifdef GRAPHICS
        if (fmod(T, 250 * dt) < dt) {
            clearScreen();

            updateBuffers();
            draw();

//            for (ContainerObject * c : objs) {
//                c -> updateBuffers();
//                c -> draw();
//            }

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

//    for (ContainerObject * c : objs) {
//        c -> generateBuffers();
//    }

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
        this -> colors = colorbuffer;
    }

    {
        GLuint elementbuffer; // create buffer for main cube object
        glGenBuffers(1, &elementbuffer);
        this -> indices = elementbuffer;
    }

    {
        GLuint vertexbuffer;
        glGenBuffers(1, &vertexbuffer); // bind cube vertex buffer
        this -> vertices = vertexbuffer;
    }
}

void Simulation::updateBuffers() {
    {
        GLfloat color_buffer_data[3 * masses.size()];

        for (int i = 0; i < masses.size(); i++) {
            color_buffer_data[3 * i] = (GLfloat) mass_arr[i].color[0];
            color_buffer_data[3 * i + 1] = (GLfloat) mass_arr[i].color[1];
            color_buffer_data[3 * i + 2] = (GLfloat) mass_arr[i].color[2];
        }

        glBindBuffer(GL_ARRAY_BUFFER, colors);
        glBufferData(GL_ARRAY_BUFFER, sizeof(color_buffer_data), color_buffer_data, GL_STATIC_DRAW);
    }

    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->indices);

        GLuint indices[2 * springs.size()]; // this contains the order in which to draw the lines between points
        for (int i = 0; i < springs.size(); i++) {
            indices[2 * i] = (springs[i]->_left) -> arrayptr - mass_arr;
            indices[2 * i + 1] = (springs[i]->_right)->arrayptr - mass_arr;
        }

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * springs.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW); // second argument is number of bytes
    }

    {
        GLfloat vertex_data[3 * masses.size()];

        for (int i = 0; i < masses.size(); i++) {
            vertex_data[3 * i] = (GLfloat) mass_arr[i].getPosition()[0];
            vertex_data[3 * i + 1] = (GLfloat) mass_arr[i].getPosition()[1];
            vertex_data[3 * i + 2] = (GLfloat) mass_arr[i].getPosition()[2];
        }

        glBindBuffer(GL_ARRAY_BUFFER, vertices);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);
    }
}

void Simulation::draw() {
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, this -> vertices);
    glPointSize(10);
    glLineWidth(10);
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

Plane * Simulation::createPlane(const Vec & abc, double d ) { // creates half-space ax + by + cz < d
    Plane * new_plane = new Plane(abc, d);
    constraints.push_back(new_plane);
    return new_plane;
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