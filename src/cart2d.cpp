#include "cart2d.hpp"
#include <cassert>
#include "log2.hpp"
#include <string>

std::map<std::string,int> varxIds;

cart2d_func::cart2d_func(struct inputConfig &cf_) : rk_func(cf_) {
  var   = FS4D("var", cf.ngi, cf.ngj, cf.ngk, cf.nvt);     // Primary Variables
  tmp1  = FS4D("tmp1", cf.ngi, cf.ngj, cf.ngk, cf.nvt);    // Temporary Array
  dvar  = FS4D("dvar", cf.ngi, cf.ngj, cf.ngk, cf.nvt);    // RHS Output
  vel   = FS3D("vel", cf.ngi, cf.ngj, 2);                  // velocity
  p     = FS2D("p", cf.ngi, cf.ngj);                       // Pressure
  T     = FS2D("T", cf.ngi, cf.ngj);                       // Temperature
  rho   = FS2D("rho", cf.ngi, cf.ngj);                     // Total Density
  fluxx = FS2D("fluxx", cf.ngi, cf.ngj); // Advective Fluxes in X direction
  fluxy = FS2D("fluxy", cf.ngi, cf.ngj); // Advective Fluxes in Y direction

  varNames.push_back("X-Momentum");
  varNames.push_back("Y-Momentum");
  varNames.push_back("Energy");
  for (int v=0; v<cf.ns; ++v)
      varNames.push_back(cf.speciesName[v]+" Density");

  assert(varNames.size() == cf.nvt);

  varxNames.push_back("X-Velocity");
  varxNames.push_back("Y-Velocity");
  varxNames.push_back("Pressure");
  varxNames.push_back("Temperature");
  varxNames.push_back("Total Density");

  if (cf.ceq){
    varxNames.push_back("C_dvar_u");
    varxNames.push_back("C_dvar_v");
  }


  varx  = FS4D("varx", cf.ngi, cf.ngj, cf.ngk, varxNames.size()); // Extra Vars

  // Create and copy minimal configuration array for data needed
  // withing Kokkos kernels.
  cd = FS1D("deviceCF", 6 + cf.ns * 3);
  FS1DH hostcd = Kokkos::create_mirror_view(cd);
  Kokkos::deep_copy(hostcd, cd);
  hostcd(0) = cf.ns; // number of gas species
  hostcd(1) = cf.dx; // cell size
  hostcd(2) = cf.dy; // cell size
  hostcd(3) = cf.dz; // cell size
  hostcd(4) = cf.nv; // number of flow variables
  hostcd(5) = cf.ng; // number of flow variables
 
  // include gas properties for each gas species
  int sdx = 6;
  for (int s = 0; s < cf.ns; ++s) {
    hostcd(sdx) = cf.gamma[s];        // ratio of specific heats
    hostcd(sdx + 1) = cf.R / cf.M[s]; // species gas comstant
    hostcd(sdx + 2) = cf.mu[s];       // kinematic viscosity
    sdx += 3;
  }
  Kokkos::deep_copy(cd, hostcd); // copy congifuration array to device

  // Create Simulation )timers
  timers["advect"] = Timer::fiestaTimer("Advection Term Calculation");
  timers["pressgrad"] = Timer::fiestaTimer("Pressure Gradient Calculation");
  timers["calcSecond"] = Timer::fiestaTimer("Secondary Variable Calculation");
  timers["solWrite"] = Timer::fiestaTimer("Solution Write Time");
  timers["resWrite"] = Timer::fiestaTimer("Restart Write Time");
  timers["statCheck"] = Timer::fiestaTimer("Status Check");
  timers["rk"] = Timer::fiestaTimer("Runge Stage Update");
  timers["halo"] = Timer::fiestaTimer("Halo Exchanges");
  timers["bc"] = Timer::fiestaTimer("Boundary Conditions");
  if (cf.visc) {
    timers["visc"] = Timer::fiestaTimer("Viscous Term Calculation");
  }
  if (cf.ceq) {
    timers["ceq"] = Timer::fiestaTimer("C-Equation");
  }
};

void cart2d_func::preSim() {
  timers["calcSecond"].reset();
  policy_f ghost_pol = policy_f({0, 0}, {cf.ngi, cf.ngj});
  Kokkos::parallel_for(ghost_pol, calculateRhoPT2D(var, p, rho, T, cd));
  Kokkos::parallel_for(ghost_pol, computeVelocity2D(var, rho, vel));
  Kokkos::parallel_for(ghost_pol, copyExtraVars2D(varx, vel, p, rho, T));
  Kokkos::fence();
  timers["calcSecond"].accumulate();
}

void cart2d_func::preStep() {
}

void cart2d_func::compute() {
  policy_f ghost_pol = policy_f({0, 0}, {cf.ngi, cf.ngj});
  policy_f cell_pol  = policy_f({cf.ng, cf.ng}, {cf.ngi - cf.ng, cf.ngj - cf.ng});
  policy_f face_pol  = policy_f({cf.ng - 1, cf.ng - 1}, {cf.ngi - cf.ng, cf.ngj - cf.ng});

  // Calcualte Total Density and Pressure Fields
  timers["calcSecond"].reset();
  Kokkos::parallel_for(ghost_pol, calculateRhoPT2D(var, p, rho, T, cd));
  Kokkos::parallel_for(ghost_pol, computeVelocity2D(var, rho, vel));
  Kokkos::fence();
  timers["calcSecond"].accumulate();

  // Calculate and apply weno fluxes for each variable
  for (int v = 0; v < cf.nv; ++v) {
    timers["advect"].reset();
    if (cf.scheme == 3) {
      Kokkos::parallel_for( face_pol, computeFluxQuick2D(var, p, rho, fluxx, fluxy, cd, v));
    } else if (cf.scheme == 2) {
      Kokkos::parallel_for( face_pol, computeFluxCentered2D(var, p, rho, fluxx, fluxy, cd, v));
    } else {
      Kokkos::parallel_for( face_pol, computeFluxWeno2D(var, p, rho, vel, fluxx, fluxy, cf.dx, cf.dy, v));
    }
    //Kokkos::fence();
    Kokkos::parallel_for(cell_pol, advect2D(dvar, fluxx, fluxy, cd, v));
    Kokkos::fence();
    timers["advect"].accumulate();
  }

  // Apply Pressure Gradient Term
  timers["pressgrad"].reset();
  Kokkos::parallel_for(cell_pol, applyPressureGradient2D(dvar, p, cd));
  Kokkos::fence();
  timers["pressgrad"].accumulate();
}

void cart2d_func::postStep() {
  bool varsxNeeded=false;

  if ( (cf.write_freq >0) && ((cf.t+1) % cf.write_freq == 0) ) varsxNeeded=true;
  if ( (cf.stat_freq  >0) && ((cf.t+1) % cf.stat_freq  == 0) ) varsxNeeded=true;
  if (cf.ioThisStep) varsxNeeded = true;

  // compute secondary variables
  if (varsxNeeded){
    timers["calcSecond"].reset();
    policy_f ghost_pol = policy_f({0, 0}, {cf.ngi, cf.ngj});
    Kokkos::parallel_for(ghost_pol, calculateRhoPT2D(var, p, rho, T, cd));
    Kokkos::parallel_for(ghost_pol, computeVelocity2D(var, rho, vel));
    Kokkos::parallel_for(ghost_pol, copyExtraVars2D(varx, vel, p, rho, T));
    Kokkos::fence();
    timers["calcSecond"].accumulate();
  }

}

void cart2d_func::postSim() {
}
