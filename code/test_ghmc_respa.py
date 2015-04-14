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

positions = lb_loader.pre_equil(system, positions, temperature)

#groups = [(0, 4), (1, 2), (2, 1)]
groups = [(0, 1), (1, 1), (2, 1)]
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


"""
# 4 2 1
            Enew           Eold  accept  acceptance_rate    deltaE            ke  naccept  ntrials
0  247736.062500  247735.109375       1         0.614919  0.953125  63250.007812      305      496
1  247747.593750  247748.484375       1         0.615694 -0.890625  63305.453125      306      497
2  247779.796875  247775.359375       0         0.614458  4.437500  63812.046875      306      498
3  247763.515625  247760.218750       1         0.615230  3.296875  63264.609375      307      499
4  247775.390625  247769.406250       0         0.614000  5.984375  63666.125000      307      500
"""
