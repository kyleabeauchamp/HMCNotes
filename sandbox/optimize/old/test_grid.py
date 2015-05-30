import lb_loader
import sklearn.grid_search
import scipy.stats.distributions
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 1.0 / u.picoseconds
n_steps = 500
temperature = 300. * u.kelvin


unigen = lambda min, max: scipy.stats.uniform(loc=min, scale=(max - min))
min_timestep = 0.75
max_timestep = 1.5


params_grid = {
"steps_per_hmc" : scipy.stats.distributions.randint(low=10, high=30),
"timestep" : unigen(min_timestep, max_timestep)
}



def hmc_inner(system, positions, params):
    integrator = hmc_integrators.GHMC2(temperature, params["steps_per_hmc"], params["timestep"] * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(1)  # timings will not accumulate on first step
    integrator.step(n_steps)
    d = integrator.summary
    d.update(params)
    print(pd.Series(d))
    return d


def optimize_hmc(system, positions):
    data = pd.DataFrame([hmc_inner(system, positions, params) for params in params_list])
    return data


#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
#system, positions = testsystem.system, testsystem.positions

system, positions = lb_loader.load_lb()
#hmc_integrators.guess_force_groups(system)
positions = lb_loader.pre_equil(system, positions, temperature)

n_iter = 30
params_list = list(sklearn.grid_search.ParameterSampler(params_grid, n_iter=n_iter))
data = optimize_hmc(system, positions)
