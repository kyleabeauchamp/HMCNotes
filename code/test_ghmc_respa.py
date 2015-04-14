import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds
timestep = 1.0 * u.femtoseconds
steps_per_hmc = 12

system, positions = lb_loader.load_lb()
integrators.guess_force_groups(system)

groups = [(0, 4), (1, 2), (2, 1)]
integrator = integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

for i in range(100):
    data = []
    for j in range(5):
        integrator.step(1)
        data.append(integrator.summary())
    data = pd.DataFrame(data)
    print(data)

