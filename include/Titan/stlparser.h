//
// Created by Jacob Austin on 5/26/18.
//

#ifndef STL_PARSER_H
#define STL_PARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace stl {
    class Vec3D {
    public:
        Vec3D(); // default
        Vec3D(const Vec3D & v); // copy constructor
        Vec3D(double x, double y, double z); // initialization from x, y, and z values
        Vec3D & operator = (const Vec3D & v);
        Vec3D & operator += (const Vec3D & v);
        Vec3D operator - () const;
        double & operator [] (int n);
        const double & operator [] (int n) const;

        friend Vec3D operator + (const Vec3D & x, const Vec3D & y);
        friend Vec3D operator - (const Vec3D & x, const Vec3D & y);

        friend Vec3D operator * (const Vec3D & v, const double x); // double and Vec
        friend Vec3D operator * (const double x, const Vec3D & v);
        friend Vec3D operator * (const Vec3D & v1, const Vec3D & v2); // two Vecs (elementwise)

        friend Vec3D operator / (const Vec3D & v, const double x); // double and vec
        friend Vec3D operator / (const Vec3D & v1, const Vec3D & v2); // two Vecs (elementwise)

        friend std::ostream & operator << (std::ostream &, const Vec3D &); // print

        double norm() const; // gives vector norm
        double sum() const; // gives vector norm

    private:
        double data[3] = { 0 }; // initialize data to 0
    };

    double dot(const Vec3D & a, const Vec3D & b);
    Vec3D cross(const Vec3D & v1, const Vec3D & v2);


    Vec3D::Vec3D() {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
    }

    Vec3D::Vec3D(double x, double y, double z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }

    Vec3D::Vec3D(const Vec3D & v) {
        data[0] = v.data[0];
        data[1] = v.data[1];
        data[2] = v.data[2];
    }

    Vec3D & Vec3D::operator=(const Vec3D & v) {
        if (this == &v) {
            return *this;
        }

        data[0] = v.data[0];
        data[1] = v.data[1];
        data[2] = v.data[2];

        return *this;
    }

    Vec3D & Vec3D::operator+=(const Vec3D & v) {
        data[0] += v.data[0];
        data[1] += v.data[1];
        data[2] += v.data[2];
        return *this;
    }

    Vec3D Vec3D::operator-() const{
        return Vec3D(-data[0], -data[1], -data[2]);
    }


    Vec3D operator+(const Vec3D & v1, const Vec3D & v2) {
        return Vec3D(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1], v1.data[2] + v2.data[2]);
    }

    Vec3D operator-(const Vec3D & v1, const Vec3D & v2) {
        return Vec3D(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1], v1.data[2] - v2.data[2]);
    }

    Vec3D operator*(const double x, const Vec3D & v) {
        return Vec3D(v.data[0] * x, v.data[1] * x, v.data[2] * x);
    }

    Vec3D operator*(const Vec3D & v, const double x) {
        return x * v;
    }

    Vec3D operator*(const Vec3D & v1, const Vec3D & v2) {
        return Vec3D(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1], v1.data[2] * v2.data[2]);
    }

    Vec3D operator/(const Vec3D & v, const double x) {
        return Vec3D(v.data[0] / x, v.data[1] / x, v.data[2] / x);
    }

    Vec3D operator/(const Vec3D & v1, const Vec3D & v2) {
        return Vec3D(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1], v1.data[2] / v2.data[2]);
    }

    std::ostream & operator << (std::ostream & strm, const Vec3D & v) {
        return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    }

    double & Vec3D::operator [] (int n) {
        if (n < 0 || n >= 3) {
            std::cerr << std::endl << "Out of bounds" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            return data[n];
        }
    }

    const double & Vec3D::operator [] (int n) const {
        if (n < 0 || n >= 3) {
            std::cerr << std::endl << "Out of bounds" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            return data[n];
        }
    }

    double Vec3D::norm() const {
        return sqrt(pow(data[0], 2) + pow(data[1], 2) + pow(data[2], 2));
    }

    double Vec3D::sum() const {
        return data[0] + data[1] + data[2];
    }

    double dot(const Vec3D & a, const Vec3D & b) {
        return (a * b).sum();
    }

    Vec3D cross(const Vec3D & v1, const Vec3D & v2) {
        return Vec3D(v1[1] * v2[2] - v1[2] * v2[1], v2[0] * v1[2] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
    }

    struct Triangle {
        Triangle(const Vec3D &n, const Vec3D &vertex1, const Vec3D &vertex2, const Vec3D &vertex3) : normal(n), v1(vertex1),
                                                                                             v2(vertex2),
                                                                                             v3(vertex3) {};

        friend std::ostream & operator<<(std::ostream &, Triangle & t);

        Vec3D normal;
        Vec3D v1;
        Vec3D v2;
        Vec3D v3;
    };

    struct BBox {
        BBox(const Vec3D & center, double x, double y, double z) {
            this -> center = center;
            this -> xdim = x;
            this -> ydim = y;
            this -> zdim = z;
        }

        Vec3D center;
        double xdim, ydim, zdim;
    };

    struct stlFile {
        std::string header;
        int num_triangles;
        std::vector<Triangle> triangles;

        BBox getBoundingBox();
        bool inside(const Vec3D & point, int num_rays = 10);
    };

        BBox stlFile::getBoundingBox() {
            double min[3] = {DBL_MIN};
            double max[3] = {DBL_MIN};

            for (Triangle t : triangles) {
                for (int i = 0; i < 3; i++) {
                    min[i] = fmin(t.v1[i], min[i]);
                    max[i] = fmax(t.v1[i], max[i]);

                    min[i] = fmin(t.v2[i], min[i]);
                    max[i] = fmax(t.v2[i], max[i]);

                    min[i] = fmin(t.v3[i], min[i]);
                    max[i] = fmax(t.v3[i], max[i]);
                }
            }

            return BBox(Vec3D((max[0] - min[0]) / 2 + min[0], (max[1] - min[1]) / 2 + min[1], (max[2] - min[2]) / 2 + min[2]), max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        }

        bool intersect(const Vec3D & point, const Vec3D & ray, const Triangle & t, double EPSILON) {
            Vec3D edge1 = t.v2 - t.v1;
            Vec3D edge2 = t.v3 - t.v1;
            Vec3D h = cross(ray, edge2);
            double a = dot(edge1, h);

//            std::cout << "a is: " << a << std::endl;

            if (a > -EPSILON && a < EPSILON)
                return false;

            double f = 1 / a;
            Vec3D s = point - t.v1;
            double u = f * dot(s, h);

            if (u < 0 || u > 1.0)
                return false;

//            std::cout << "u is: " << u << std::endl;

            Vec3D q = cross(s, edge1);
            double v = f * dot(ray, q);

            if (v < 0 || u + v > 1.0)
                return false;

//            std::cout << "v is: " << v << std::endl;

            if (f * dot(edge2, q) > EPSILON)
                return true;
            else
                return false;
        }

        double randDouble(double min, double max) {
            return min + (double) rand() / RAND_MAX * (max - min);
        }

        bool stlFile::inside(const Vec3D & point, int num_rays) {
            int count = 0;
            int ray_count = 0;

            const double EPSILON = 0.000001;

            for (int i = 0; i < num_rays; i++) {
                Vec3D ray = Vec3D(randDouble(-1000, 1000), randDouble(-1000, 1000), randDouble(-1000, 1000));
                ray = ray / ray.norm();

                for (Triangle t : triangles) {
                   if (intersect(point, ray, t, EPSILON)) {
                       count++;
                   }
                }

                if (count % 2 == 1) {
                    ray_count++;
                }

                count = 0;
            }

//            std::cout << "ray_count is: " << ray_count << std::endl;

            double fraction = (double) ray_count / (double) num_rays;

//            std::cout << "frac is: " << fraction << std::endl;

            if (fraction > 0.5) {
                return true;
            } else {
                return false;
            }
        }

    Vec3D parseVec(std::ifstream & file) {
        char triangle[12];
        file.read(triangle, 12);
        return Vec3D(*(float *) (triangle), *(float *) (triangle + 4), *(float *) (triangle + 8));
    }

    std::ostream & operator<<(std::ostream &strm, Triangle & t) {
        strm << "Normal : " << t.normal << std::endl;
        strm << "Vertex 1 : " << t.v1 << std::endl;
        strm << "Vertex 2 : " << t.v2 << std::endl;
        strm << "Vertex 3 : " << t.v3 << std::endl;
        return strm;
    }

    stlFile parseSTL(std::string path) {
        std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);

        if (!file) {
            std::cerr << "ERROR: COULD NOT READ FILE." << std::endl;
            assert(0);
        } else {
            std::cout << "File found. Parsing STL file." << std::endl;
        }

        char header[80];
        char num_triangles[4];
        file.read(header, 80);
        file.read(num_triangles, 4);

        stlFile data;
        data.header = std::string(header);
        data.num_triangles = * (unsigned int *) num_triangles;

        for (int i = 0; i < data.num_triangles; i++) {
            Vec3D normal = parseVec(file);
            Vec3D v1 = parseVec(file);
            Vec3D v2 = parseVec(file);
            Vec3D v3 = parseVec(file);
            data.triangles.push_back(Triangle(normal, v1, v2, v3));

            char properties[2];
            file.read(properties, 2);
        }

        std::cout << "Found " << data.num_triangles << " triangles. Parsing complete!" << std::endl;

        assert((file.peek(), file.eof()));

        return data;
    }
}

#endif //STL_PARSER_H
