import lb_loader
import itertools
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds
temperature = 25. * u.kelvin

testsystem = testsystems.LennardJonesFluid()

system, positions = testsystem.system, testsystem.positions

positions = np.loadtxt("./sandbox/ljbox.dat")
length = 2.66723326712
boxes = ((length, 0, 0), (0, length, 0), (0, 0, length))
system.setDefaultPeriodicBoxVectors(*boxes)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.5 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(20000)
positions = context.getState(getPositions=True).getPositions()


timestep = 2 * u.femtoseconds  # LJ Cluster
n_iter = 160000

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 2.)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

data, t0, g, Neff = lb_loader.converge(context, n_steps=100, Neff_cutoff=300, sleep_time=10)
