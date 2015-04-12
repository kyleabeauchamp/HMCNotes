import sklearn.grid_search
import scipy.stats.distributions
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

collision_rate = 1.0 / u.picoseconds
n_steps = 5000
temperature = 300. * u.kelvin


unigen = lambda min, max: scipy.stats.uniform(loc=min, scale=(max - min))
min_timestep = 1.0
max_timestep = 2.25
n_iter = 20

params_grid = {
"steps_per_hmc" : scipy.stats.distributions.randint(low=10, high=20),
"timestep" : unigen(min_timestep, max_timestep)
}
params_list = list(sklearn.grid_search.ParameterSampler(params_grid, n_iter=n_iter))

def hmc_inner(system, positions, params):
    print(params)
    integrator = integrators.GHMC2(temperature, params["steps_per_hmc"], params["timestep"] * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    score = integrator.effective_timestep
    print(integrator.acceptance_rate, score)
    return score

def optimize_hmc(system, positions):
    scores = [hmc_inner(system, positions, params) for params in params_list]
    return scores


testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
system = testsystem.system

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(testsystem.system, integrator)
context.setPositions(testsystem.positions)
context.setVelocitiesToTemperature(temperature)
integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()

scores = optimize_hmc(system, positions)

"""
In [29]: max(scores)
Out[29]: Quantity(value=1.5050494824835965, unit=femtosecond)

{'steps_per_hmc': 11, 'timestep': 2.139677967704857}
"""
