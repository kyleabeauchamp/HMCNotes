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


for hydrogenMass in [1.0 * u.amu, 2.0 * u.amu, 3.0 * u.amu, 4.0 * u.amu]:

    system, positions = lb_loader.load_lb(hydrogenMass=hydrogenMass)

    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
    context = mm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(10000)
    positions = context.getState(getPositions=True).getPositions()

    collision_rate = 1.0 / u.picoseconds
    timestep = 1.5 * u.femtoseconds
    steps_per_hmc = 12
    k_max = 3

    integrator = integrators.GHMC2(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    integrator.step(2500)
    data = integrator.vstep(5)
