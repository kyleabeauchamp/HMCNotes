import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"

sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, use_hmc=False)



timestep = 1.0 * u.femtoseconds
steps_per_hmc = 25

integrator = hmc_integrators.MJHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
f = lambda x: integrator.getGlobalVariableByName(x)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

integrator.step(1)

print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
print(f("last_move"))
#print(f("ELm"), f("E0"), f("E"))
#print(f("KLm"), f("K0"), f("K"))
print(f("ELm") + f("KLm"), f("E0") + f("K0"), f("E") + f("K"))
print(f("gammaL"), f("gammaF"), f("gammaR"))
print(f("wL"), f("wF"), f("wR"))