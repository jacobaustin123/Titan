#ifndef LOCH_OBJECT_H
#define LOCH_OBJECT_H

#include <vector>
#include <thrust/device_vector.h>

#include "mass.h"
#include "spring.h"
#include "vec.h"

#ifdef GRAPHICS
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/device_vector.h>

#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif


__device__ const double NORMAL = 100000;

class Constraint { // constraint like plane or sphere which applies force to masses
public:
    virtual ~Constraint() {};

#ifdef GRAPHICS
    bool _initialized;
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
#endif
};

//class CudaBall : public Constraint { // ball constraint, force is inversely proportional to distance
//public:
//    CudaBall(const Vec & center, double r);
//    void setRadius(double r) { _radius = r; }
//    void setCenter(const Vec & center) { _center = center; }
//    Vec getForce(const Vec & position);
//    void translate(const Vec & displ);
//
//#ifdef GRAPHICS
//    virtual ~CudaBall() {
//        //glDeleteBuffers(1, &vertices);
//        //glDeleteBuffers(1, &colors);
//    }
//
//    void generateBuffers();
//    void draw();
//
//    void subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth);
//    void writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3);
//    void tangentize(GLfloat * v);
//
//    int depth = 2;
//
//    GLuint vertices;
//    GLuint colors;
//#endif
//
//    double _radius;
//    Vec _center;
//};
//
//
//class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
//public:
//    Plane() {};
//    Plane(const Vec & tangent, double d);
//
//    Vec getForce(const Vec & position);
//    void translate(const Vec & displ);
//
//    void settangent(const Vec & tangent) { _tangent = tangent; }; // tangent is (a, b, c)
//    void setOffset(double d) { _offset = d; }; // ax + by + cz < d
//
//    Vec _tangent;
//    double _offset;
//
//#ifdef GRAPHICS
//    virtual ~Plane() {
//        glDeleteBuffers(1, &vertices);
//        glDeleteBuffers(1, &colors);
//    }
//
//    void generateBuffers();
//    void draw();
//
//    GLuint vertices;
//    GLuint colors;
//#endif
//};

struct Ball : public Constraint {
    Ball(const Vec & center, double radius) {
        _center = center;
        _radius = radius;

#ifdef GRAPHICS
        _initialized = false;
#endif

    }

    double _radius;
    Vec _center;

#ifdef GRAPHICS
    ~Ball() {
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
#endif
};

struct CudaBall {
    CUDA_CALLABLE_MEMBER CudaBall() = default;
    CUDA_CALLABLE_MEMBER CudaBall(const Vec & center, double radius) {
        _center = center;
        _radius = radius;
    }

    CUDA_CALLABLE_MEMBER CudaBall(const Ball & b) {
        _center = b._center;
        _radius = b._radius;
    }

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m) {
        double dist = (m -> pos - _center).norm();
        m -> force += (dist <= _radius) ? NORMAL * (m -> pos - _center) / dist : Vec(0, 0, 0);
    }

    double _radius;
    Vec _center;
};

struct ContactPlane : public Constraint {
    ContactPlane(const Vec & normal, double offset) {
        _normal = normal / normal.norm();
        _offset = offset;

#ifdef GRAPHICS
        _initialized = false;
#endif
    }

    Vec _normal;
    double _offset;

#ifdef GRAPHICS
    ~ContactPlane() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    GLuint vertices;
    GLuint colors;
#endif
};

struct CudaContactPlane {
    CUDA_CALLABLE_MEMBER CudaContactPlane() = default;

    CUDA_CALLABLE_MEMBER CudaContactPlane(const Vec & normal, double offset) {
        _normal = normal / normal.norm();
        _offset = offset;
    }

    CudaContactPlane(const ContactPlane & p) {
        _normal = p._normal;
        _offset = p._offset;
    }

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m) {
        double disp = dot(m -> pos, _normal) - _offset;
        m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host
    }

    Vec _normal;
    double _offset;
};


struct CudaConstraintPlane {
    CUDA_CALLABLE_MEMBER CudaConstraintPlane(const Vec & normal, double friction) {
        _normal = normal / normal.norm();
        _friction = friction;
    }

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m) {
        m -> vel += - _normal * dot(m -> vel, _normal); // constraint velocity

        double normal_force = dot(m -> force, _normal);
        m -> force += - _normal * normal_force; // constraint force
        m -> force += - _friction * normal_force * (m -> vel) / (m -> vel).norm(); // apply friction force
    }

    Vec _normal;
    double _friction;
};

struct CudaDirection {
    CUDA_CALLABLE_MEMBER CudaDirection(const Vec & tangent, double friction) {
        _tangent = tangent / tangent.norm();
        _friction = friction;
    }

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m) {
        m -> vel = _tangent * dot(m -> vel, _tangent);

        Vec normal_force = m -> force - dot(m -> force, _tangent) * _tangent;
        m -> force += -normal_force;

        m -> force += - normal_force.norm() * _friction * _tangent;
    }

    Vec _tangent;
    double _friction;
};

struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane * d_planes;
    CudaBall * d_balls;

    int num_planes;
    int num_balls;
};


//struct LOCAL_CONSTRAINTS {
//    LOCAL_CONSTRAINTS() {
//        constraint_plane = thrust::device_vector<CudaConstraintPlane>(1);
//        contact_plane = thrust::device_vector<CudaContactPlane>(1);
//        ball = thrust::device_vector<CudaBall>(1);
//        direction = thrust::device_vector<CudaDirection>(1);
//
//        drag_coefficient = 0;
//        fixed = false;
//    }
//
//    thrust::device_vector<CudaContactPlane> contact_plane;
//    thrust::device_vector<CudaConstraintPlane> constraint_plane;
//    thrust::device_vector<CudaBall> ball;
//    thrust::device_vector<CudaDirection> direction;
//
//    int drag_coefficient;
//    bool fixed; // move here from the class itself;
//};
//
//struct CUDA_LOCAL_CONSTRAINTS {
//    CUDA_LOCAL_CONSTRAINTS(const LOCAL_CONSTRAINTS & c) {
//        contact_plane = thrust::raw_pointer_cast(c.contact_plane.data());
//        constraint_plane = thrust::raw_pointer_cast(c.constraint_plane.data());
//        ball = thrust::raw_pointer_cast(c.ball.data());
//        direction = thrust::raw_pointer_cast(c.direction.data());
//
//        num_contact_planes = c.contact_plane.size();
//        num_constraint_planes = c.constraint_plane.size();
//        num_balls = c.ball.size();
//        num_directions = c.direction.size();
//
//        fixed = c.fixed;
//        drag_coefficient = c.drag_coefficient;
//    }
//
//    CudaContactPlane * contact_plane;
//    CudaConstraintPlane * constraint_plane;
//    CudaBall * ball;
//    CudaDirection * direction;
//
//    int drag_coefficient;
//    bool fixed; // move here from the class itself;
//
//    int num_contact_planes;
//    int num_balls;
//    int num_constraint_planes;
//    int num_directions; // if this is greater than 1, just make it fixed
//};








class Container { // contains and manipulates groups of masses and springs
public:
    virtual ~Container() {};
    virtual void translate(const Vec & displ) = 0; // translate all masses by fixed amount

    void setMassValue(double m); // set masses for all Mass objects
    void setKValue(double k); // set k for all Spring objects
    void setDeltaTValue(double m); // set masses for all Mass objects
    void setRestLengthValue(double len); // set masses for all Mass objects
    void makeFixed();

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Cube : public Container {
public:
    ~Cube() {};

    Cube(const Vec & center, double side_length = 1.0);

    void translate(const Vec & displ);

    double _side_length;
    Vec _center;
};

class Lattice : public Container {
public:
    ~Lattice() {};

    Lattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    void translate(const Vec & displ);

    int nx, ny, nz;
    Vec _center, _dims;
};

#endif //LOCH_OBJECT_H