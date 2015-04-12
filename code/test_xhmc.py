import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds
timestep = 2.0 * u.femtoseconds
steps_per_hmc = 12
k_max = 2

testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
system = testsystem.system
positions = testsystem.positions


integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()

integrator = integrators.XHMCIntegrator(temperature, collision_rate, timestep, steps_per_hmc, k_max)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


for i in range(1000):
    data = []
    for j in range(10):
        integrator.step(1)
        data.append(integrator.summary())
    data = pd.DataFrame(data)
    print(data)


