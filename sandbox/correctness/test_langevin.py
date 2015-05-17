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

sysname = "switchedljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem = lb_loader.load(sysname)

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
mm.LocalEnergyMinimizer.minimize(context)
integrator.step(40000)
positions = context.getState(getPositions=True).getPositions()
print(integrator.acceptance_rate)

collision_rate = 1.0 / u.picoseconds
n_steps = 25
Neff_cutoff = 1E5

itype = "LangevinIntegrator"

integrator = mm.LangevinIntegrator(temperature, collision_rate, langevin_timestep)
context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
filename = "./data/%s_%s_%s_%.3f_%d.csv" % (precision, sysname, itype, langevin_timestep / u.femtoseconds, collision_rate * u.picoseconds)
print(filename)
integrator.step(450000)
data, start, g, Neff, mu, sigma, stderr = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff, filename=filename)
