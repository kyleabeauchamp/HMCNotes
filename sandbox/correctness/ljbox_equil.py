import pymbar
import itertools
import pandas as pd
import lb_loader
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
system.addForce(mm.MonteCarloBarostat(1.0 * u.atmospheres, temperature, 50))

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.5 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setPeriodicBoxVectors(*boxes)
context.setVelocitiesToTemperature(temperature)
integrator.step(25000)
positions = context.getState(getPositions=True).getPositions()

x = []
for i in range(1000):
    state = context.getState(getParameters=True, getPositions=True)
    positions = state.getPositions(asNumpy=True) / u.nanometers
    boxes = state.getPeriodicBoxVectors(asNumpy=True) / u.nanometers
    x.append(boxes[0][0] / u.nanometer)
    print(x[-1])
    integrator.step(1000)

length = np.mean(x)
print(length)
# 2.66723326712
boxes = ((length, 0, 0), (0, length, 0), (0, 0, length))
np.savetxt("./ljbox.dat", positions)

testsystem = testsystems.LennardJonesFluid()
testsystem.system

positions = np.loadtxt("./ljbox.dat")

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setPeriodicBoxVectors(*boxes)
context.setVelocitiesToTemperature(temperature)


for i in range(1000):
    state = context.getState(getParameters=True, getPositions=True, getEnergy=True)
    print(state.getPotentialEnergy())
    positions = state.getPositions(asNumpy=True) / u.nanometers
    integrator.step(1000)


np.savetxt("./ljbox.dat", positions)
