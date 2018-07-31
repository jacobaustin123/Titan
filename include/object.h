#ifndef LOCH_OBJECT_H
#define LOCH_OBJECT_H

//#include "mass.h"
//#include "spring.h"
#include "vec.h"

#ifdef GRAPHICS
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h> // TODO add SDL2 instead

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#endif

#include <vector>
#include <thrust/device_vector.h>


struct CUDA_MASS;
class Spring;
class Mass;

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
    CudaBall() = default;
    CUDA_CALLABLE_MEMBER CudaBall(const Vec & center, double radius);
    CUDA_CALLABLE_MEMBER CudaBall(const Ball & b);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

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
    CudaContactPlane() = default;
    CUDA_CALLABLE_MEMBER CudaContactPlane(const Vec & normal, double offset);
    CudaContactPlane(const ContactPlane & p);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _normal;
    double _offset;
};


struct CudaConstraintPlane {
    CudaConstraintPlane() = default;

    CUDA_CALLABLE_MEMBER CudaConstraintPlane(const Vec & normal, double friction);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _normal;
    double _friction;
};

struct CudaDirection {
    CudaDirection() = default;

    CUDA_CALLABLE_MEMBER CudaDirection(const Vec & tangent, double friction);

    CUDA_CALLABLE_MEMBER void applyForce(CUDA_MASS * m);

    Vec _tangent;
    double _friction;
};

struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane * d_planes;
    CudaBall * d_balls;

    int num_planes;
    int num_balls;
};


#ifdef CONSTRAINTS
struct LOCAL_CONSTRAINTS {
    LOCAL_CONSTRAINTS();

    thrust::device_vector<CudaContactPlane> contact_plane;
    thrust::device_vector<CudaConstraintPlane> constraint_plane;
    thrust::device_vector<CudaBall> ball;
    thrust::device_vector<CudaDirection> direction;

    CudaContactPlane * contact_plane_ptr;
    CudaConstraintPlane * constraint_plane_ptr;
    CudaBall * ball_ptr;
    CudaDirection * direction_ptr;

    int num_contact_planes;
    int num_balls;
    int num_constraint_planes;
    int num_directions; // if this is greater than 1, just make it fixed

    int drag_coefficient;
    bool fixed; // move here from the class itself;
};

struct CUDA_LOCAL_CONSTRAINTS {
    CUDA_LOCAL_CONSTRAINTS() = default;

    CUDA_LOCAL_CONSTRAINTS(LOCAL_CONSTRAINTS & c);

    CudaContactPlane * contact_plane;
    CudaConstraintPlane * constraint_plane;
    CudaBall * ball;
    CudaDirection * direction;

    int drag_coefficient;
    bool fixed; // move here from the class itself;

    int num_contact_planes;
    int num_balls;
    int num_constraint_planes;
    int num_directions; // if this is greater than 1, just make it fixed
};

#endif



class Container { // contains and manipulates groups of masses and springs
public:
    virtual ~Container() {};
    void translate(const Vec & displ); // translate all masses by fixed amount

    void setMassValues(double m); // set masses for all Mass objects
    void setSpringConstants(double k); // set k for all Spring objects
    void setDeltaT(double m); // set masses for all Mass objects
    void setRestLengths(double len); // set masses for all Mass objects

    void makeFixed();

    void add(Mass * m);
    void add(Spring * s);
    void add(Container * c);

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Cube : public Container {
public:
    ~Cube() {};

    Cube(const Vec & center, double side_length = 1.0);

    double _side_length;
    Vec _center;
};

class Lattice : public Container {
public:
    ~Lattice() {};

    Lattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    int nx, ny, nz;
    Vec _center, _dims;
};

#endif //LOCH_OBJECT_H