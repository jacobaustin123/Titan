CC  = clang++
CXX = clang++

INCLUDES = -I/Users/JAustin/Desktop/glfw/include -I/System/Library/Frameworks/OpenGL.framework/Headers -I./common2.1.0/include 

CFLAGS   = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES) -std=c++11

LDFLAGS = -g -L/Users/JAustin/Desktop/glfw/build/src -L/System/Library/Frameworks/OpenGL.framework/Libraries
LDLIBS  = -lglfw -lGLEW -lGL

executables = main
objects = main.o cube.o vec.o common/shader.o

.PHONY: default
default: $(executables)

$(executables): cube.o vec.o common/shader.o

$(objects): cube.hpp vec.hpp common/shader.hpp

.PHONY: clean
clean:
	rm -f *~ a.out core $(objects) $(executables)
	rm -rf *.dSYM

.PHONY: all
all: clean default
