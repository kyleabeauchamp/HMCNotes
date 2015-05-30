import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

n_steps = 3000
temperature = 300. * u.kelvin

testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
system = testsystem.system

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()

def test_hmc(timestep, steps_per_hmc, collision_rate):
    timestep = timestep * u.femtoseconds
    integrator = hmc_integrators.GHMC2(temperature, steps_per_hmc, timestep, collision_rate / u.picoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    return integrator.acceptance_rate

timestep = 2.0
collision_rate_list = np.logspace(-1, 4, 10)
steps_per_hmc = 12
data = []
for i, collision_rate in enumerate(collision_rate_list):
    print(i, timestep, collision_rate)
    acceptance = test_hmc(timestep, steps_per_hmc, collision_rate)
    data.append(dict(collision_rate=collision_rate, acceptance=acceptance))
    print(data[-1])
        
data = pd.DataFrame(data)

data["exparg"] = timestep * data.collision_rate * 1E-3  # Convert fs to ps
data["b"] = np.exp(-data.exparg)
