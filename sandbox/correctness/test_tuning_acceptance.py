import pickle
import os
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "ljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem = lb_loader.load(sysname)

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
context.getState(getEnergy=True).getPotentialEnergy()
mm.LocalEnergyMinimizer.minimize(context)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.step(300)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.acceptance_rate


collision_rate = 1.0 / u.picoseconds
n_steps = 25

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(250)
output = integrator.vstep(25)
