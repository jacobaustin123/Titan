#ifdef GRAPHICS
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>

#include "shader.h"

GLuint LoadShaders(){
    std::string FragmentShaderCode = "#version 330 core\n"
                                     "\n"
                                     "// Interpolated values from the vertex shaders\n"
                                     "in vec3 fragmentColor;\n"
                                     "\n"
                                     "// Ouput data\n"
                                     "out vec3 color;\n"
                                     "\n"
                                     "void main(){\n"
                                     "\n"
                                     "\t// Output color = color specified in the vertex shader, \n"
                                     "\t// interpolated between all 3 surrounding vertices\n"
                                     "\tcolor = fragmentColor;\n"
                                     "\n"
                                     "}";

    std::string VertexShaderCode = "#version 330 core\n"
                                   "\n"
                                   "// Input vertex data, different for all executions of this shader.\n"
                                   "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                                   "layout(location = 1) in vec3 vertexColor;\n"
                                   "\n"
                                   "// Output data ; will be interpolated for each fragment.\n"
                                   "out vec3 fragmentColor;\n"
                                   "// Values that stay constant for the whole mesh.\n"
                                   "uniform mat4 MVP;\n"
                                   "\n"
                                   "void main(){\t\n"
                                   "\n"
                                   "\t// Output position of the vertex, in clip space : MVP * position\n"
                                   "\tgl_Position =  MVP * vec4(vertexPosition_modelspace,1);\n"
                                   "\n"
                                   "\t// The color of each vertex will be interpolated\n"
                                   "\t// to produce the color of each fragment\n"
                                   "\tfragmentColor = vertexColor;\n"
                                   "}\n";

    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    GLint Result = GL_FALSE;
    int InfoLogLength;

    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
        glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    printf("Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> ProgramErrorMessage(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }


    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

#endif