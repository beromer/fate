#ifndef CART2D_H
#define CART2D_H

#include "kokkosTypes.hpp"
#include "input.hpp"
#include "rkfunction.hpp"
#include "bc.hpp"
#include "kernels/advect.hpp"
#include "kernels/flux.hpp"
#include "kernels/presgrad.hpp"
#include "kernels/secondary.hpp"
#include "kernels/velocity.hpp"

class cart2d_func : public rk_func {

public:
  cart2d_func(struct inputConfig &cf_);

  void compute();
  void preStep();
  void postStep();
  void preSim();
  void postSim();

  FS2D p;       // Pressure
  FS3D vel;     // Velocity
  FS2D T;       // Temperature
  FS2D rho;     // Total Density
  FS2D qx;      // Heat Fluxes in X direction
  FS2D qy;      // Heat Fluxes in X direction
  FS2D fluxx;   // Weno Fluxes in X direction
  FS2D fluxy;   // Weno Fluxes in Y direction
  FS4D stressx; // stress tensor on x faces
  FS4D stressy; // stress tensor on y faces
  FS3D gradRho; // Density Gradient array
  FS5D m;
  FS2D_I noise; // Noise indicator array
};

#endif
