// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// Include shaders and support files
#include "common/shader.h"
#include "vec.h"
#include "sim.h"

// #cmakedefine SOURCE_DIR "${SOURCE_DIR}"

int main()
{
    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( 1024, 768, "CUDA Physics Simulation", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glEnable(GL_DEPTH_TEST);
    //    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

//    const char * dir = SOURCE_DIR;
//    char shader1[4096];
//    strcpy(shader1, dir);
//    strcat(shader1, "/shaders/TransformVertexShader.vertexshader");
//
//    char shader2[4096];
//    strcpy(shader2, dir);
//    strcat(shader2, "/shaders/ColorFragmentShader.fragmentshader");

    // Create and compile our GLSL program from the shaders
    //    GLuint programID = LoadShaders( "SimpleTransform.vertexshader", "SingleColor.fragmentshader" );
    GLuint programID = LoadShaders("shaders/TransformVertexShader.vertexshader", "shaders/ColorFragmentShader.fragmentshader");
    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");

    // Projection matrix : 45∞ Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
    // Or, for an ortho camera :
    //glm::mat4 Projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates

    // Camera matrix
    glm::mat4 View = glm::lookAt(
                       glm::vec3(15, 15, 10), // Camera is at (4,3,3), in World Space
                       glm::vec3(0,0,8), // and looks at the origin
                       glm::vec3(0,0,1)  // Head is up (set to 0,-1,0 to look upside-down)
                       );
    // Model matrix : an identity matrix (model will be at the origin)
    glm::mat4 Model = glm::mat4(1.0f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around

    GLuint vertexbuffer;
    int count = 0;
    GLfloat g_vertex_buffer_data[24];

    GLuint vertexbufferplatform; // this contains the vertices for the base (constraint) platform
    GLfloat vertex_buffer_platform[108] = {
        -0.3f,-0.3f,-0.02f,
        -0.3f,-0.3f,0.0f,
        -0.3f, 0.3f,0.0f,
        0.3f, 0.3f,-0.02f,
        -0.3f,-0.3f,-0.02f,
        -0.3f, 0.3f,-0.02f,
        0.3f,-0.3f,0.0f,
        -0.3f,-0.3f,-0.02f,
        0.3f,-0.3f,-0.02f,
        0.3f, 0.3f,-0.02f,
        0.3f,-0.3f,-0.02f,
        -0.3f,-0.3f,-0.02f,
        -0.3f,-0.3f,-0.02f,
        -0.3f, 0.3f, 0.0f,
        -0.3f, 0.3f,-0.02f,
        0.3f,-0.3f, 0.0f,
        -0.3f,-0.3f, 0.0f,
        -0.3f,-0.3f,-0.02f,
        -0.3f, 0.3f, 0.0f,
        -0.3f,-0.3f, 0.0f,
        0.3f,-0.3f, 0.0f,
        0.3f, 0.3f, 0.0f,
        0.3f,-0.3f,-0.02f,
        0.3f, 0.3f,-0.02f,
        0.3f,-0.3f,-0.02f,
        0.3f, 0.3f, 0.0f,
        0.3f,-0.3f, 0.0f,
        0.3f, 0.3f, 0.0f,
        0.3f, 0.3f,-0.02f,
        -0.3f, 0.3f,-0.02f,
        0.3f, 0.3f, 0.0f,
        -0.3f, 0.3f,-0.02f,
        -0.3f, 0.3f, 0.0f,
        0.3f, 0.3f, 0.0f,
        -0.3f, 0.3f, 0.0f,
        0.3f,-0.3f, 0.0f
    };

    glGenBuffers(1, &vertexbufferplatform); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertexbufferplatform);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_platform), vertex_buffer_platform, GL_STATIC_DRAW);

    static const GLfloat g_color_buffer_data[] = { // color buffer (green) for these constraint vertices
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
        0.5f, 1.0f, 0.2f,
    };

    GLuint colorbuffer; // generate color buffer
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

    static const GLfloat g_color_buffer_data_2[] = { // colors for the cube
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
        1.0f, 0.2f, 0.2f,
    };

    GLuint colorbuffer2; // bind cube colors to buffer colorbuffer2
    glGenBuffers(1, &colorbuffer2);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data_2), g_color_buffer_data_2, GL_STATIC_DRAW);

    std::vector<GLubyte> indices; // this contains the order in which to draw the lines between points
    for (int i = 0; i < 7; i++) {
        for (int j = i + 1; j < 8; j++) {
            indices.push_back(i);
            indices.push_back(j);
        }
    }

    GLuint elementbuffer; // create buffer for main cube object
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size(), &indices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

    Simulation sim; // initialize simulation object

    Cube * c = sim.createCube(Vec(0, 0, 10), 2.0); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(1000); // set the spring constant for all springs to 10
    c -> setMassValue(2.0); // set all masses to 2.0
    c -> setDeltaTValue(0.0001); // set the dt value for all masses in the cube to 0.00001
    c -> setRestLengthValue(3.0); // set the rest length of all springs to

    sim.createPlane(Vec(0, 0, 1), 0);

    sim.setBreakpoint(0.03);
    sim.run();

    do{
        usleep(2000);

//        std::cout << c -> masses[0]->getPosition()[0] << std::endl;
        for (int i = 0; i < 8; i++) { // populate buffer with position data for cube vertices
            g_vertex_buffer_data[3 * i] = (GLfloat) c -> masses[i] -> getPosition()[0];
            g_vertex_buffer_data[3 * i + 1] = (GLfloat) c -> masses[i]->getPosition()[1];
            g_vertex_buffer_data[3 * i + 2] = (GLfloat) c -> masses[i]->getPosition()[2];
        }

        count++; // iterate count

//        if (count % 1000 == 0) // print some random information every 1000 iterations
//            std::cout << count << ": " << c ->masses[0] -> getPosition() << std::endl;

        glGenBuffers(1, &vertexbuffer); // bind cube vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

        // Use our shader
        glUseProgram(programID);

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        // 1st attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbufferplatform);

        glVertexAttribPointer(
              0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
              3,                  // size
              GL_FLOAT,           // type
              GL_FALSE,           // normalized?
              0,                  // stride
              (void*)0            // array buffer offset
              );

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
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
        //
        //
        //        // Draw the triangle !
        //        glDrawArrays(GL_TRIANGLES, 0, 36); // 3 indices starting at 0 -> 1 triangle

        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
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
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer2);
        glVertexAttribPointer(
                              1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
                              3,                                // size
                              GL_FLOAT,                         // type
                              GL_FALSE,                         // normalized?
                              0,                                // stride
                              (void*)0                          // array buffer offset
                              );


        glDrawArrays(GL_POINTS, 0, 8); // 3 indices starting at 0 -> 1 triangle
        glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_BYTE, (void*) 0); // 3 indices starting at 0 -> 1 triangle

        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(0);


        // Swap buffers
        glfwSwapInterval(0);
        glfwSwapBuffers(window);
        glfwPollEvents();

        sim.setBreakpoint(sim.time() + 0.01);
        sim.resume();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 && sim.time() < 100.0 );

            // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteProgram(programID);
    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}
