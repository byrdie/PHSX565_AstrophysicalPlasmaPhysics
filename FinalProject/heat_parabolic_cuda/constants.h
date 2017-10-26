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
const uint ly = 1;
const uint lz = 1;

// size of strides
const uint sx = lx - m;
const uint sy = 1;
const uint sz = 1;




// number of strides
const uint Nx = 1024;
const uint Ny = 1;
const uint Nz = 1;

// Size of domain in gridpoints (including boundary cells)
const uint Lt = 32;
const uint Lx = Nx * sx + m;
const uint Ly = Ny * sy + m;
const uint Lz = Nz * sz;
const uint L = Lx * Ly * Lz * Lt;

// Size of the domain in bytes
const uint Lt_B = Lt * sizeof(float);
const uint Lx_B = Lx * sizeof(float);
const uint Ly_B = Ly * sizeof(float);
const uint Lz_B = Lz * sizeof(float);
const uint L_B = L * sizeof(float);

// Specify the size of the domain in physical units
const float Dx = 1.0;
const float Dy = 1.0;
const float Dz = 1.0;

// Calculate the spatial step size
const float dx = Dx / (float) Lx;
const float dy = Dy / (float) Ly;
const float dz = Dz / (float) Lz;

// calculate the temporal step size
const float g = 1.0 / 2.0;	// factor below maximum step size
const float dt = g * (dx * dx) / (2.0 * kappa);	// CFL condition
const float Dt = dt * Lt;






#endif /* CONSTANTS_C_ */
