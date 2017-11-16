/*
 * constants.c
 *
 *  Created on: Oct 25, 2017
 *      Author: byrdie
 */

#ifndef CONSTANTS_C_
#define CONSTANTS_C_

const bool debug = true;

// boundary conditions
const float T_right = 1.0;
const float T_left = 0.1;

// define physical constants
const float kappa_max = 1.0;


// specify the order of the differentiation in each direction
const uint m_f = 1;
const uint m_b = 1;
const uint m = m_b + m_f;

// size of stride
const uint lx = 5;

// number of strides
const uint Nx = 20;

// buffer size
const uint bt = 2;

// downsampling factor
const uint wt = 1;

// final size after downsampling
const uint Wt =  1024;

// Size of domain in gridpoints (including boundary cells)
const uint Lt = Wt * wt;
//const uint Lx = Nx * sx + m;
const uint Lx = Nx * lx;
const uint L = Lx * Wt;



// Specify the size of the domain in physical units
const float Dx = 1.0;

// Calculate the spatial step size
const float dx = Dx / (float) (Lx - 1);

// calculate the parabolic step size
const float f_CFL = 1.0 / 4.0;	// factor below maximum step size
const float dt_p = f_CFL * (dx * dx) / (2.0 * kappa_max);	// CFL condition

const float beta = 10.0;

// calculate the hyperbolic step size
const float dt_h =  beta * dt_p;

// calculate hyperbolic propagation speed
const float c_h = f_CFL * dx / dt_h;

//float g = 1e-4;
//
//float dt_h = f_CFL * dx * sqrt(g);
//float c_h = f_CFL * dx / dt_h;




#endif /* CONSTANTS_C_ */
