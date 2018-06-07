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

static double DISPL_CONST = 10000;

class BaseObject { // base class for larger objects like Cubes, etc.
public:
    virtual void translate(const Vec & displ) = 0; // translate all masses by fixed amount
};

class Constraint : public BaseObject { // constraint like plane or sphere which applies force to masses
public:
    virtual ~Constraint() {};

    virtual Vec getForce(const Vec & position) = 0; // returns force on an object based on its position, e.g. plane or
#ifdef GRAPHICS
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
#endif
};

class ContainerObject : public BaseObject { // contains and manipulates groups of masses and springs
public:
    void setMassValue(double m); // set masses for all Mass objects
    void setKValue(double k); // set k for all Spring objects
    void setDeltaTValue(double m); // set masses for all Mass objects
    void setRestLengthValue(double len); // set masses for all Mass objects
    void makeFixed();

    // we can have more of these
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

//class Ball : public Constraint { // ball constraint, force is inversely proportional to distance
//public:
//    void setRadius(double r) { _radius = r; }
//    void setCenter(const Vec & center) { _center = center; }
//    Vec getForce(const Vec & position);
//
//#ifdef GRAPHICS
//    void generateBuffers();
//    void draw();
//
//    GLuint vertices;
//    GLuint colors;
//#endif
//
//    double _radius;
//    Vec _center;
//};

class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
public:
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

class Cube : public ContainerObject {
public:
    Cube(const Vec & center, double side_length = 1.0);

    void translate(const Vec & displ);

    double _side_length;
    Vec _center;
};

class Lattice : public ContainerObject {
public:
    Lattice(const Vec & center, const Vec & dims, int nx = 10, int ny = 10, int nz = 10);

    void translate(const Vec & displ);

    int nx, ny, nz;
    Vec _center, _dims;
};

#endif //LOCH_OBJECT_H