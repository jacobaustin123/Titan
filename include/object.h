#ifndef LOCH_OBJECT_H
#define LOCH_OBJECT_H

#include <vector>
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

class Base { // base class for larger objects like Cubes, etc.
public:
    virtual void translate(const Vec & displ) = 0; // translate all masses by fixed amount
};

class Constraint : public Base { // constraint like plane or sphere which applies force to masses
public:
    virtual ~Constraint() {};

    virtual Vec getForce(const Vec & position) = 0; // returns force on an object based on its position, e.g. plane or
#ifdef GRAPHICS
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
#endif
};

class Container : public Base { // contains and manipulates groups of masses and springs
public:
    virtual ~Container() {};

    void setMassValue(double m); // set masses for all Mass objects
    void setKValue(double k); // set k for all Spring objects
    void setDeltaTValue(double m); // set masses for all Mass objects
    void setRestLengthValue(double len); // set masses for all Mass objects
    void makeFixed();

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Ball : public Constraint { // ball constraint, force is inversely proportional to distance
public:
    Ball(const Vec & center, double r);
    void setRadius(double r) { _radius = r; }
    void setCenter(const Vec & center) { _center = center; }
    Vec getForce(const Vec & position);
    void translate(const Vec & displ);

#ifdef GRAPHICS
    virtual ~Ball() {
        //glDeleteBuffers(1, &vertices);
        //glDeleteBuffers(1, &colors);
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

    double _radius;
    Vec _center;
};


struct CUDA_BALL {
    CUDA_CALLABLE_MEMBER CUDA_BALL() = default;
    CUDA_CALLABLE_MEMBER CUDA_BALL(const Ball & b) { _radius = b._radius; _center = b._center; }
    CUDA_CALLABLE_MEMBER Vec getForce(const Vec & position) {
        double dist = (position - _center).norm();
        return (dist <= _radius) ? NORMAL * (position - _center) / dist : Vec(0, 0, 0);
    }

    double _radius;
    Vec _center;
};

class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
public:
    Plane() {};
    Plane(const Vec & normal, double d);

    Vec getForce(const Vec & position);
    void translate(const Vec & displ);

    void setNormal(const Vec & normal) { _normal = normal; }; // normal is (a, b, c)
    void setOffset(double d) { _offset = d; }; // ax + by + cz < d

    Vec _normal;
    double _offset;

#ifdef GRAPHICS
    virtual ~Plane() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    GLuint vertices;
    GLuint colors;
#endif
};

struct CUDA_PLANE {
    CUDA_CALLABLE_MEMBER CUDA_PLANE() = default;
    CUDA_CALLABLE_MEMBER CUDA_PLANE(const Plane & p) { _normal = p._normal; _offset = p._offset; }

    CUDA_CALLABLE_MEMBER Vec getForce(const Vec & position) {
        double disp = dot(position, _normal) - _offset;
        return (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host
    }

    Vec _normal;
    double _offset;
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

struct CUDA_CONSTRAINT_STRUCT {
    CUDA_PLANE * d_planes;
    CUDA_BALL * d_balls;

    int num_planes;
    int num_balls;
};

#endif //LOCH_OBJECT_H