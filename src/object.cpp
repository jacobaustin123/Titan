//
// Created by Jacob Austin on 5/21/18.
//

#include "object.h"
#include <cmath>
#include "sim.h"


#ifdef GRAPHICS
// Include GLEW
#include <GL/glew.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#endif

#ifdef GRAPHICS
struct GraphicsBall {
    GraphicsBall() = default;

    GraphicsBall(const Vec & center, double radius) {
        _center = center;
        _radius = radius;
    }

    ~GraphicsBall() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    void subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth);
    void writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3);
    void normalize(GLfloat * v);

    int depth = 2;

    GLuint vertices;
    GLuint colors;

    double _radius;
    Vec _center;
};
#endif

#ifdef GRAPHICS
struct GraphicsPlane {
    GraphicsPlane() = default;

    GraphicsPlane(const Vec & normal, double offset) {
        _normal = normal;
        _offset = offset;
    }

    ~GraphicsPlane() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    GLuint vertices;
    GLuint colors;

    double _offset;
    Vec _normal;
};
#endif


Vec Plane::getForce(const Vec & position) { // returns force on an object based on its position, e.g. plane or
    double disp = dot(position, _normal) - _offset;
//    if (disp < 0) printf("%.15e\n", round(- disp * DISPL_CONST * _normal, 4)[2]);
    return (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // - disp
}

Plane::Plane(const Vec & normal, double d) {
    _offset = d;
    _normal = normal / normal.norm();

#ifdef GRAPHICS
    gplane = new GraphicsPlane(_normal, _offset);
#endif
}

void Plane::translate(const Vec & displ) {
    _offset += dot(displ, _normal);
}

void Ball::translate(const Vec & displ) {
    _center += displ;
}

Ball::Ball(const Vec & center, double r) {
    _center = center;
    _radius = r;
#ifdef GRAPHICS
    gball = new GraphicsBall(center, r);
#endif
}

#ifdef GRAPHICS
void Ball::generateBuffers() {
    gball -> generateBuffers();
}

void Ball::draw() {
    gball -> draw();
}

void Plane::generateBuffers() {
    gplane -> generateBuffers();
}

void Plane::draw() {
    gplane -> draw();
}

#endif

void ContainerObject::setMassValue(double m) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> setMass(m);
    }
}

void ContainerObject::setKValue(double k) {
    for (Spring * spring : springs) {
        spring -> setK(k);
    }
}

void ContainerObject::setDeltaTValue(double dt) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> setDeltaT(dt);
    }
}

void ContainerObject::setRestLengthValue(double len) { // set masses for all Mass objects
    for (Spring * spring : springs) {
        spring -> setRestLength(len);
    }
}

void ContainerObject::makeFixed() {
    for (Mass * mass : masses) {
        mass -> makeFixed();
    }
}


Cube::Cube(const Vec & center, double side_length) {
    _center = center;
    _side_length = side_length;

    for (int i = 0; i < 8; i++) {
        masses.push_back(new Mass(side_length * (Vec(i & 1, (i >> 1) & 1, (i >> 2) & 1) - Vec(0.5, 0.5, 0.5)) + center));
    }

    for (int i = 0; i < 8; i++) { // add the appropriate springs
        for (int j = i + 1; j < 8; j++) {
            springs.push_back(new Spring(masses[i], masses[j]));
        }
    }
}

void Cube::translate(const Vec & displ) {
    for (Mass * m : masses) {
        m->translate(displ);
    }
}

Beam::Beam(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    _center = center;
    _dims = dims;
    this -> nx = nx;
    this -> ny = ny;
    this -> nz = nz;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                masses.push_back(new Mass(Vec((nx > 1) ? (double) i / (nx - 1.0) - 0.5 : 0, (ny > 1) ? j / (ny - 1.0) - 0.5 : 0, (nz > 1) ? k / (nz - 1.0) - 0.5 : 0) * dims + center));
                if (i == 0) {
                    masses[masses.size() - 1] -> fixed = true;
                }
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int l = 0; l < ((i != nx - 1) ? 2 : 1); l++) {
                    for (int m = 0; m < ((j != ny - 1) ? 2 : 1); m++) {
                        for (int n = 0; n < ((k != nz - 1) ? 2 : 1); n++) {
                            if (l != 0 || m != 0 || n != 0) {
                                springs.push_back(new Spring(masses[k + j * nz + i * ny * nz],
                                                             masses[(k + n) + (j + m) * nz + (i + l) * ny * nz]));
                            }
                        }
                    }
                }

                if (k != nz - 1) {
                    if (j != ny - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz], // get the full triangle
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                    }

                    if (i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }

                    if (j != ny - 1 && i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + (j + 1) * nz + (i + 1) * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + (i + 1) * ny * nz],
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + (j + 1) * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }
                }

                if (j != ny - 1 && i != nx - 1) {
                    springs.push_back(new Spring(masses[k + (j + 1) * nz + i * ny * nz],
                                                 masses[k + j * nz + (i + 1) * ny * nz]));
                }
            }
        }
    }
}

void Beam::translate(const Vec &displ) {
    for (Mass * m : masses) {
        m -> pos += displ;
    }
}

Lattice::Lattice(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    _center = center;
    _dims = dims;
    this -> nx = nx;
    this -> ny = ny;
    this -> nz = nz;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                masses.push_back(new Mass(Vec((nx > 1) ? (double) i / (nx - 1.0) - 0.5 : 0, (ny > 1) ? j / (ny - 1.0) - 0.5 : 0, (nz > 1) ? k / (nz - 1.0) - 0.5 : 0) * dims + center));
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int l = 0; l < ((i != nx - 1) ? 2 : 1); l++) {
                    for (int m = 0; m < ((j != ny - 1) ? 2 : 1); m++) {
                        for (int n = 0; n < ((k != nz - 1) ? 2 : 1); n++) {
                            if (l != 0 || m != 0 || n != 0) {
                                springs.push_back(new Spring(masses[k + j * nz + i * ny * nz],
                                                             masses[(k + n) + (j + m) * nz + (i + l) * ny * nz]));
                            }
                        }
                    }
                }

                if (k != nz - 1) {
                    if (j != ny - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz], // get the full triangle
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                    }

                    if (i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }

                    if (j != ny - 1 && i != nx - 1) {
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + i * ny * nz],
                                                     masses[k + (j + 1) * nz + (i + 1) * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + j * nz + (i + 1) * ny * nz],
                                                     masses[k + (j + 1) * nz + i * ny * nz]));
                        springs.push_back(new Spring(masses[(k + 1) + (j + 1) * nz + i * ny * nz],
                                                     masses[k + j * nz + (i + 1) * ny * nz]));
                    }
                }

                if (j != ny - 1 && i != nx - 1) {
                    springs.push_back(new Spring(masses[k + (j + 1) * nz + i * ny * nz],
                                                 masses[k + j * nz + (i + 1) * ny * nz]));
                }
            }
        }
    }
}

void Lattice::translate(const Vec &displ) {
    for (Mass * m : masses) {
        m -> pos += displ;
    }
}

Vec Ball::getForce(const Vec & position) {
    double dist = (position - _center).norm();
//    std::cout << dist << std::endl;
    return (dist <= _radius) ? NORMAL * (position - _center) / dist : Vec(0, 0, 0);
}

#ifdef GRAPHICS

Ball::~Ball() { delete gball; }

void GraphicsBall::normalize(GLfloat * v) {
    GLfloat norm = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2],2)) / _radius;

    for (int i = 0; i < 3; i++) {
        v[i] /= norm;
    }
}

void GraphicsBall::writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3) {
    for (int j = 0; j < 3; j++) {
        arr[j] = v1[j] + _center[j];
    }

    arr += 3;


    for (int j = 0; j < 3; j++) {
        arr[j] = v2[j] + _center[j];
    }

    arr += 3;

    for (int j = 0; j < 3; j++) {
        arr[j] = v3[j] + _center[j];
    }
}

void GraphicsBall::subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth) {
    GLfloat v12[3], v23[3], v31[3];

    if (depth == 0) {
        writeTriangle(arr, v1, v2, v3);
        return;
    }

    for (int i = 0; i < 3; i++) {
        v12[i] = v1[i]+v2[i];
        v23[i] = v2[i]+v3[i];
        v31[i] = v3[i]+v1[i];
    }

    normalize(v12);
    normalize(v23);
    normalize(v31);

    subdivide(arr, v1, v12, v31, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v2, v23, v12, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v3, v31, v23, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v12, v23, v31, depth - 1);
}


void GraphicsBall::generateBuffers() {
    glm::vec3 color = {0.22f, 0.71f, 0.0f};

    GLfloat * vertex_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // times 4 for subdivision

    GLfloat X = (GLfloat) _radius * .525731112119133606;
    GLfloat Z = (GLfloat) _radius * .850650808352039932;

    static GLfloat vdata[12][3] = {
            {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
            {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
            {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
    };
    static GLuint tindices[20][3] = {
            {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
            {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
            {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
            {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

    for (int i = 0; i < 20; i++) {
        subdivide(&vertex_data[3 * 3 * (int) pow(4, depth) * i], vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], depth);
    }


    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), vertex_data, GL_STATIC_DRAW);

    GLfloat * color_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // TODO constant length array

    for (int i = 0; i < 20 * 3 * (int) pow(4, depth); i++) {
        color_data[3*i] = color[0];
        color_data[3*i + 1] = color[1];
        color_data[3*i + 2] = color[2];
    }

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), color_data, GL_STATIC_DRAW);

    delete [] color_data;
    delete [] vertex_data;
}

void GraphicsBall::draw() {
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 20 * 3 * (int) pow(4, depth)); // 12*3 indices starting at 0 -> 12 triangles

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

#endif

#ifdef GRAPHICS

Plane::~Plane() { delete gplane; }

void GraphicsPlane::generateBuffers() {
    glm::vec3 color = {0.22f, 0.71f, 0.0f};

    std::cout << "normal: " << _normal << std::endl;

    std::cout << cross(Vec(0, 0, 1), Vec(0, 1, 0)) << std::endl;

    Vec temp = (dot(_normal, Vec(0, 1, 0)) < 0.8) ? Vec(0, 1, 0) : Vec(1, 0, 0);

    Vec v1 = cross(_normal, temp); // two unit vectors along plane
    v1 = v1 / v1.norm();

    Vec v2 = cross(_normal, v1);
    v2 = v2 / v2.norm();

    std::cout << "v1: " << v1 << " v2: " << v2 << std::endl;

    const static GLfloat vertex_buffer_platform[118] = {
            -1, -1, -1,
            -1, -1,  1,
            -1,  1,  1,
            1,  1, -1,
            -1, -1, -1,
            -1,  1, -1,
            1, -1,  1,
            -1, -1, -1,
            1, -1, -1,
            1,  1, -1,
            1, -1, -1,
            -1, -1, -1,
            -1, -1, -1,
            -1,  1,  1,
            -1,  1, -1,
            1, -1,  1,
            -1, -1,  1,
            -1, -1, -1,
            -1,  1,  1,
            -1, -1,  1,
            1, -1,  1,
            1,  1,  1,
            1, -1, -1,
            1,  1, -1,
            1, -1, -1,
            1,  1,  1,
            1, -1,  1,
            1,  1,  1,
            1,  1, -1,
            -1,  1, -1,
            1,  1,  1,
            -1,  1, -1,
            -1,  1,  1,
            1,  1,  1,
            -1,  1,  1,
            1, -1,  1
    };

    GLfloat vertex_data[108];

    for (int i = 0; i < 36; i++) {
        Vec temp = Vec(vertex_buffer_platform[3 * i], vertex_buffer_platform[3 * i + 1], vertex_buffer_platform[3 * i + 2]);
        Vec vertex = 10 * dot(v1, temp) * v1 + 10 * dot(v2, temp) * v2 + _normal * (_offset + dot(_normal, temp) - 1.0);

        std::cout << vertex << std::endl;

        vertex_data[3 * i] = vertex[0];
        vertex_data[3 * i + 1] = vertex[1];
        vertex_data[3 * i + 2] = vertex[2];
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);

    GLfloat g_color_buffer_data[108];

    for (int i = 0; i < 36; i++) {
        g_color_buffer_data[3 * i] = color[0];
        g_color_buffer_data[3 * i + 1] = color[1];
        g_color_buffer_data[3 * i + 2] = color[2];
    }


    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);
}

void GraphicsPlane::draw() {
    // 1st attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}
#endif