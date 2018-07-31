//
// Created by Jacob Austin on 5/21/18.
//

#include "object.h"
#include <cmath>
#include "sim.h"

CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Vec & center, double radius) {
    _center = center;
    _radius = radius;
}

CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Ball & b) {
    _center = b._center;
    _radius = b._radius;
}

CUDA_CALLABLE_MEMBER void CudaBall::applyForce(CUDA_MASS * m) {
    double dist = (m -> pos - _center).norm();
    m -> force += (dist <= _radius) ? NORMAL * (m -> pos - _center) / dist : Vec(0, 0, 0);
}

CUDA_CALLABLE_MEMBER CudaContactPlane::CudaContactPlane(const Vec & normal, double offset) {
    _normal = normal / normal.norm();
    _offset = offset;
}

CudaContactPlane::CudaContactPlane(const ContactPlane & p) {
    _normal = p._normal;
    _offset = p._offset;
}

CUDA_CALLABLE_MEMBER void CudaContactPlane::applyForce(CUDA_MASS * m) {
    double disp = dot(m -> pos, _normal) - _offset;
    m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host
}

CUDA_CALLABLE_MEMBER CudaConstraintPlane::CudaConstraintPlane(const Vec & normal, double friction) {
    assert(normal.norm() != 0.0);

    _normal = normal / normal.norm();
    _friction = friction;
}

CUDA_CALLABLE_MEMBER void CudaConstraintPlane::applyForce(CUDA_MASS * m) {
    double normal_force = dot(m -> force, _normal);
    m -> force += - _normal * normal_force; // constraint force

    if (m -> vel.norm() != 0.0) {
        m -> vel += - _normal * dot(m -> vel, _normal); // constraint velocity
        m -> force += - _friction * normal_force * (m -> vel) / (m -> vel).norm(); // apply friction force
    }
}

CUDA_CALLABLE_MEMBER CudaDirection::CudaDirection(const Vec & tangent, double friction) {
    assert(tangent.norm() != 0.0);

    _tangent = tangent / tangent.norm();
    _friction = friction;
}

CUDA_CALLABLE_MEMBER void CudaDirection::applyForce(CUDA_MASS * m) {
    Vec normal_force = m -> force - dot(m -> force, _tangent) * _tangent;
    m -> force += - normal_force;

    if (m -> vel.norm() != 0.0) {
        m -> vel = _tangent * dot(m -> vel, _tangent);
        m -> force += - normal_force.norm() * _friction * _tangent;
    }
}

void Container::setMassValues(double m) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> m += m;
    }
}

void Container::setSpringConstants(double k) {
    for (Spring * spring : springs) {
        spring -> _k = k;
    }
}

void Container::setDeltaT(double dt) { // set masses for all Mass objects
    for (Mass * mass : masses) {
        mass -> dt += dt;
    }
}

void Container::setRestLengths(double len) { // set masses for all Mass objects
    for (Spring * spring : springs) {
        spring -> _rest = len;
    }
}

void Container::add(Mass * m) {
    masses.push_back(m);
}

void Container::add(Spring * s) {
    springs.push_back(s);
}

void Container::add(Container * c) {
    for (Mass * m : c -> masses) {
        masses.push_back(m);
    }

    for (Spring * s : c -> springs) {
        springs.push_back(s);
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

    for (Spring * s : springs) {
        s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
    }
}

void Container::translate(const Vec & displ) {
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

    for (Spring * s : springs) {
        s -> setRestLength((s -> _right -> pos - s -> _left -> pos).norm());
    }
}

#ifdef CONSTRAINTS

void Container::makeFixed() {
    for (Mass * mass : masses) {
        mass -> constraints.fixed = true;
    }
}

LOCAL_CONSTRAINTS::LOCAL_CONSTRAINTS() {
//    constraint_plane = thrust::device_vector<CudaConstraintPlane>(1);
//    contact_plane = thrust::device_vector<CudaContactPlane>(1);
//    ball = thrust::device_vector<CudaBall>(1);
//    direction = thrust::device_vector<CudaDirection>(1);
//
//    contact_plane_ptr = thrust::raw_pointer_cast(contact_plane.data()); // TODO make sure this is safe
//    constraint_plane_ptr = thrust::raw_pointer_cast(constraint_plane.data());
//    ball_ptr = thrust::raw_pointer_cast(ball.data());
//    direction_ptr = thrust::raw_pointer_cast(direction.data());

    num_contact_planes = 0;
    num_constraint_planes = 0;
    num_balls = 0;
    num_directions = 0;

    drag_coefficient = 0;
    fixed = false;
}

CUDA_LOCAL_CONSTRAINTS::CUDA_LOCAL_CONSTRAINTS(LOCAL_CONSTRAINTS & c) {
    contact_plane = c.contact_plane_ptr;
    constraint_plane = c.constraint_plane_ptr;
    ball = c.ball_ptr;
    direction = c.direction_ptr;

    num_contact_planes = c.num_contact_planes;
    num_constraint_planes = c.num_constraint_planes;
    num_balls = c.num_balls;
    num_directions = c.num_directions;

    fixed = c.fixed;
    drag_coefficient = c.drag_coefficient;
}

#endif

#ifdef GRAPHICS

void Ball::normalize(GLfloat * v) {
    GLfloat norm = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2],2)) / _radius;

    for (int i = 0; i < 3; i++) {
        v[i] /= norm;
    }
}

void Ball::writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3) {
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

void Ball::subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth) {
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


void Ball::generateBuffers() {
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

    _initialized = true;
}

void Ball::draw() {
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

void ContactPlane::generateBuffers() {

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

    _initialized = true;
}

void ContactPlane::draw() {
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