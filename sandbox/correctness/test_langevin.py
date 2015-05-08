import os
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

precision = "double"

sysname = "ho"

system, positions, groups, temperature, timestep0 = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep0 / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50000)
positions = context.getState(getPositions=True).getPositions()

collision_rate = 1.0 / u.picoseconds
n_steps = 25
Neff_cutoff = 40000.

itype = "LangevinIntegrator"

for timestep_factor in [1.0, 2.0, 4.0, 8.0]:
    timestep = (timestep0 / timestep_factor)
    integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
    filename = "./data/%s_%s_%s_%.3f_%d.csv" % (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
    
    if os.path.exists(filename):
        continue
    
    print(filename)
    data, start, g, Neff = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff)
    data.to_csv(filename)
    
