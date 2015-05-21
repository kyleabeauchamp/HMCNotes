import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "chargedswitchedaccurateljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(system, temperature, timestep, positions, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.step(2500)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.acceptance_rate
positions = context.getState(getPositions=True).getPositions()


collision_rate = 1.0 / u.picoseconds
n_steps = 25

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(250)
output = integrator.vstep(25)
