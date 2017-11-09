/*
 * constants.c
 *
 *  Created on: Oct 25, 2017
 *      Author: byrdie
 */

#ifndef CONSTANTS_C_
#define CONSTANTS_C_



// boundary conditions
const float T_left = 0.1;
const float T_right = 1.0;

// define physical constants
const float kappa_max = T_right;


// specify the order of the differentiation in each direction
const uint m_f = 1;
const uint m_b = 1;
const uint m = m_b + m_f;

// size of stride
const uint lx = 16;

// number of strides
const uint Nx = 16;

// downsampling factor
const uint wt = 1024;

// final size after downsampling
const uint Wt =  1024;

// Size of domain in gridpoints (including boundary cells)
const uint Lt = Wt;
//const uint Lx = Nx * sx + m;
const uint Lx = Nx * lx;
const uint L = Lx * Lt;



// Specify the size of the domain in physical units
const float Dx = 1.0;

// Calculate the spatial step size
const float dx = Dx / (float) Lx;

// calculate the parabolic step size
const float f_CFL = 1.0 / 2.0;	// factor below maximum step size
const float dt_p = f_CFL * (dx * dx) / (2.0 * kappa_max);	// CFL condition

// calculate the hyperbolic step size
const float dt_h =  dt_p;

// calculate hyperbolic propagation speed
const float c_h = f_CFL * dx / dt_h;




#endif /* CONSTANTS_C_ */
