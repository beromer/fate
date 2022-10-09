#include "fiesta.hpp"
#include "log2.hpp"
#include "cart2d.hpp"

int main(int argc, char *argv[]) {
  int exit_value=0;
  {
    // read input file and initialize configuration
    Fiesta::Simulation sim;
    Fiesta::initialize(sim.cf,argc,argv);

    sim.cf.totalTimer.start();
    sim.cf.initTimer.start();
 
    // Choose Module
    sim.f = std::make_unique<cart2d_func>(sim.cf);

    Fiesta::initializeSimulation(sim);

    Log::message("Executing pre-simulation hook");
    sim.f->preSim();

    sim.cf.initTimer.stop();

    Log::message("Beginning Main Time Loop");
    sim.cf.simTimer.start();
    for (int t = sim.cf.tstart; t < sim.cf.tend+1; ++t) {
      Fiesta::checkIO(sim,t);

      if (sim.cf.exitFlag==1){
        exit_value=1;
        break;
      }

      Fiesta::step(sim,t);
    }
    sim.cf.simTimer.stop();
    Log::message("Simulation complete!");
 
    Log::message("Executing post-simulation hook");
    sim.f->postSim();

    sim.cf.totalTimer.stop();
    Fiesta::reportTimers(sim.cf,sim.f);
  }
  Fiesta::finalize();
  return exit_value;
}
