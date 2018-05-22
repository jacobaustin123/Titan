////
////  cube.cpp
////  CUDA Physics
////
////  Created by Jacob Austin on 5/13/18.
////  Copyright © 2018 Jacob Austin. All rights reserved.
////
//
//#include <stdio.h>
//#include <iostream>
//#include <vector>
//#include "cube.h"
//
//Cube::Cube() { // create masses at the four corners
//    mass = 0.8;
//    Vec temp;
//    temp = Vec(0.05, 0.05, 0.2);
//    masses[0] = Mass(0.1, temp);
//    temp = Vec(0.05, -0.05, 0.2);
//    masses[1] = Mass(0.1, temp);
//    temp = Vec(-0.05, -0.05, 0.2);
//    masses[2] = Mass(0.1, temp);
//    temp = Vec(-0.05, 0.05, 0.2);
//    masses[3] = Mass(0.1, temp);
//    temp = Vec(0.05, 0.05, 0.1);
//    masses[4] = Mass(0.1, temp);
//    temp = Vec(0.05, -0.05, 0.1);
//    masses[5] = Mass(0.1, temp);
//    temp = Vec(-0.05, -0.05, 0.1);
//    masses[6] = Mass(0.1, temp);
//    temp = Vec(-0.05, 0.05, 0.1);
//    masses[7] = Mass(0.1, temp);
//
//
//    for (int i = 0; i < 8; i++) { // add the appropriate springs
//        for (int j = i + 1; j < 8; j++) {
//            springs.push_back(Spring(k, (masses[i].P - masses[j].P).norm(), &masses[i], &masses[j]));
//        }
//    }
//}