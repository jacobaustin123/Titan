#ifndef LOCH_OBJECT_H
#define LOCH_OBJECT_H

#include <vector>
#include "mass.h"
#include "spring.h"
#include "vec.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

static double DISPL_CONST = 10000;

// base class for larger objects like Cubes, etc.

class BaseObject { // base commands for all objects
public:
    virtual ~BaseObject() {};
    virtual void translate(const Vec & displ) = 0; // translate all masses by fixed amount
};

class Constraint : public BaseObject { // constraint like plane or sphere which applies force to masses
public:
    virtual Vec getForce(const Vec & position) = 0; // returns force on an object based on its position, e.g. plane or
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
};

class ContainerObject : public BaseObject { // contains and manipulates groups of masses and springs
public:

    //Set
    void setMassValue(double m); // set masses for all Mass objects
    void setKValue(double k); // set k for all Spring objects
    void setDeltaTValue(double m); // set masses for all Mass objects
    void setRestLengthValue(double len); // set masses for all Mass objects
    void makeFixed();

    // Local list of masses (under sim.masses/springs)
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;

    //Graphics
    virtual void generateBuffers() = 0;
    virtual void updateBuffers() = 0;
    virtual void draw() = 0;

    // Local list of masses (under sim.masses/springs)
    std::vector<Mass *> masses;
    std::vector<Spring *> springs;
};

class Ball : public Constraint { // ball constraint, force is inversely proportional to distance
public:
    //Set
    void setRadius(double r) { _radius = r; }
    void setCenter(const Vec & center) { _center = center; }

private:
    //Properties
    double _radius;
    Vec _center;
};

class Plane : public Constraint { // plane constraint, force is proportional to negative distance wrt plane
public:

    Plane(const Vec & normal, double d);

    //Properties
    Vec _normal;   //Normal Force [N]
    double _offset;

    //Set
    void setNormal(const Vec & normal) { _normal = normal; }; // normal is (a, b, c)
    void setOffset(double d) { _offset = d; }; // ax + by + cz < d

    //Modify
    void translate(const Vec & displ);

    //Get
    Vec getForce(const Vec & position);

    //Graphics
    void generateBuffers();
    void draw();
    GLuint vertices;
    GLuint colors;
};

class Cube : public ContainerObject {
public:
    Cube(const Vec & center, double side_length = 1.0);

    //Properties
    double _side_length;
    Vec _center;

    //Modify
    void translate(const Vec & displ);

    //Graphics
    virtual ~Cube() {
        glDeleteBuffers(1, &colors);
        glDeleteBuffers(1, &indices);
        glDeleteBuffers(1, &vertices);
    };
    void generateBuffers();
    void updateBuffers();
    void draw();

    GLuint colors;
    GLuint vertices;
    GLuint indices;
};

#endif //LOCH_OBJECT_H
