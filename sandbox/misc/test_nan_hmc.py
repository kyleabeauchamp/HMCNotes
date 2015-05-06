import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds

sysname = "water"

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(10000)
positions = context.getState(getPositions=True).getPositions()

timestep = 3.0 * u.femtoseconds

integrator = integrators.HMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(100)
output = integrator.vstep(20)
integrator.effective_timestep
