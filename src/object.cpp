//
// Created by Jacob Austin on 5/21/18.
//

#include "object.h"
#include <cmath>

inline double round( double val )
{
    if( val < 0 ) return ceil(val - 0.5);
    return floor(val + 0.5);
}

inline Vec round(const Vec & v, float n) {
    Vec temp = pow(10.0, n) * v;
    return Vec(round(temp[0]), round(temp[1]), round(temp[2])) / pow(10.0, n);
}

Vec Plane::getForce(const Vec & position) { // returns force on an object based on its position, e.g. plane or
    double disp = dot(position, _normal) - _offset;
//    if (disp < 0) printf("%.15e\n", round(- disp * DISPL_CONST * _normal, 4)[2]);
    return (disp < 0) ? - disp * DISPL_CONST * _normal : 0 * _normal; // - disp
}

Plane::Plane(const Vec & normal, double d) {
    _offset = d;
    _normal = normal;
}

void Plane::translate(const Vec & displ) {
    _offset += dot(displ, _normal);
}

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
        masses.push_back(new Mass(1.0, side_length * (Vec(i & 1, (i >> 1) & 1, (i >> 2) & 1) - Vec(0.5, 0.5, 0.5)) + center));
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

Lattice::Lattice(const Vec & center, const Vec & dims, int nx, int ny, int nz) {
    _center = center;
    _dims = dims;
    this -> nx = nx;
    this -> ny = ny;
    this -> nz = nz;

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                masses.push_back(new Mass(1.0, Vec((nx > 1) ? (double) i / (nx - 1.0) - 0.5 : 0, (ny > 1) ? j / (ny - 1.0) - 0.5 : 0, (nz > 1) ? k / (nz - 1.0) - 0.5 : 0) * dims + center));
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

//Vec Ball::getForce(const Vec & position) {
//    double dist = (position - _center).norm();
//    return (dist <= _radius) ? 10000 * (position - _center) / dist : Vec(0, 0, 0);
//}
//
//#ifdef GRAPHICS
//
//void Ball::generateBuffers() {
//    int n_phi = 10; // num points in sphere 2 * n_theta + 2 * n_theta * (n_phi - 3)
//    int n_theta = 20; // num points in sphere
//
//    GLfloat sphere_vertices[2 * n_theta + 2 * n_theta * (n_phi - 3)];
//
//    for (int i = 1; i < n_phi - 1; i++) {
//        for (int j = 0; j < n_theta; j++) {
//            sphere_vertices[n_theta * i + j] = {};
//        }
//    }
//
//}
//
//void Ball::draw() {
//    // nothing
//}
//
//#endif

#ifdef GRAPHICS

void Plane::generateBuffers() {

    float length = 5;
    float width = 5;
    float depth = 1;
    glm::vec3 color = {0.22f, 0.71f, 0.0f};

    GLfloat vertex_buffer_platform[108] = {
            -length, -width,-depth,
            -length, -width,0.0f,
            -length, width,0.0f,
            length, width,-depth,
            -length, -width,-depth,
            -length, width,-depth,
            length, -width,0.0f,
            -length, -width,-depth,
            length, -width,-depth,
            length, width,-depth,
            length, -width,-depth,
            -length, -width,-depth,
            -length, -width,-depth,
            -length, width, 0.0f,
            -length, width,-depth,
            length, -width, 0.0f,
            -length, -width, 0.0f,
            -length, -width,-depth,
            -length, width, 0.0f,
            -length, -width, 0.0f,
            length, -width, 0.0f,
            length, width, 0.0f,
            length, -width,-depth,
            length, width,-depth,
            length, -width,-depth,
            length, width, 0.0f,
            length, -width, 0.0f,
            length, width, 0.0f,
            length, width,-depth,
            -length, width,-depth,
            length, width, 0.0f,
            -length, width,-depth,
            -length, width, 0.0f,
            length, width, 0.0f,
            -length, width, 0.0f,
            length, -width, 0.0f
    };

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_platform), vertex_buffer_platform, GL_STATIC_DRAW);

    GLfloat g_color_buffer_data[] = {
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
            color[0], color[1], color[2],
    };

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);
}

void Plane::draw() {
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