/*
 * constants.c
 *
 *  Created on: Oct 25, 2017
 *      Author: byrdie
 */

#ifndef CONSTANTS_C_
#define CONSTANTS_C_

// define physical constants
const float kappa = 1.0;

// specify the order of the differentiation in each direction
const uint m_f = 1;
const uint m_b = 1;
const uint m = m_b + m_f;

// size of stride + ghost cells
const uint lx = 1024;

// size of strides
const uint sx = lx - m;

// number of strides
const uint Nx = 1024;

// Size of domain in gridpoints (including boundary cells)
const uint Lt = 16;
const uint Lx = Nx * sx + m;
//const uint Lx = Nx * lx;
const uint L = Lx * Lt;

// Size of the domain in bytes
const uint Lt_B = Lt * sizeof(float);
const uint Lx_B = Lx * sizeof(float);
const uint L_B = L * sizeof(float);

// Specify the size of the domain in physical units
const float Dx = 1.0;

// Calculate the spatial step size
const float dx = Dx / (float) Lx;

// calculate the temporal step size
const float g = 1.0 / 2.0;	// factor below maximum step size
const float dt = g * (dx * dx) / (2.0 * kappa);	// CFL condition
const float Dt = dt * Lt;






#endif /* CONSTANTS_C_ */
