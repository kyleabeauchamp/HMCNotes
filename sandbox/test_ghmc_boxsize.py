import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin

testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
system, positions = testsystem.system, testsystem.positions

rho = 0.71 ** (1 / 3.)

for box_factor in [rho ** 0, rho ** 1, rho ** 2, rho ** 3]:
    testsystem = testsystems.WaterBox(box_edge=6.0 * u.nanometers * box_factor)
    system, positions = testsystem.system, testsystem.positions
    print(system.getNumParticles())

    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(10000)
    positions = context.getState(getPositions=True).getPositions()

    collision_rate = 1.0 / u.picoseconds
    timestep = 2.5 * u.femtoseconds
    steps_per_hmc = 12
    k_max = 3

    integrator = integrators.GHMC2(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    integrator.step(2500)
    data = integrator.vstep(5)
