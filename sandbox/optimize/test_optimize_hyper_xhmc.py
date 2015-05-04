from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

collision_rate = 10000.0 / u.picoseconds

sysname = "ljbox"
n_steps = 400

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(10000)
positions = context.getState(getPositions=True).getPositions()

max_evals = 10

steps_per_hmc = hp.quniform("steps_per_hmc", 10, 25, 15)
extra_chances = hp.quniform("extra_chances", 0, 5, 5)
timestep = hp.uniform("timestep", 0.5, 2.5)

def objective(args):
    steps_per_hmc, timestep, extra_chances = args
    print(steps_per_hmc, timestep, extra_chances)
    timestep = timestep * u.femtoseconds
    integrator = integrators.XHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=extra_chances)
    context = lb_loader.build(system, integrator, positions, temperature)
    integrator.step(n_steps)
    _ = integrator.vstep(10)
    print(integrator.effective_ns_per_day)
    return integrator.effective_ns_per_day
        

space = [steps_per_hmc, timestep, extra_chances]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
