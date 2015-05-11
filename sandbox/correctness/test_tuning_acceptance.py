import os
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

precision = "single"

sysname = "switchedshortbigcluster"

system, positions, groups, temperature, timestep, testsystem = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50000)
positions = context.getState(getPositions=True).getPositions()

collision_rate = 1.0 / u.picoseconds
n_steps = 25

integrator = integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(250)
output = integrator.vstep(25)
