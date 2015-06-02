from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 200
precision = "mixed"
collision_rate = 1. / u.picoseconds

sysname = "alanine"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)

max_evals = 100

steps_per_hmc = hp.quniform("steps_per_hmc", 15, 25, 1)
extra_chances = hp.quniform("extra_chances", 0, 4, 1)
timestep = hp.uniform("timestep", 1.0, 3.25)


def inner_objective(args):
    steps_per_hmc, timestep, extra_chances = args
    print("steps=%d, timestep=%f, extra_chances=%d" % (steps_per_hmc, timestep, extra_chances))
    current_timestep = timestep * u.femtoseconds
    extra_chances = int(extra_chances)
    steps_per_hmc = int(steps_per_hmc)
    integrator = hmc_integrators.UnrolledXCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, extra_chances=extra_chances)
    simulation = lb_loader.build(testsystem, integrator, temperature)
    simulation.integrator.step(n_steps)
    return integrator

def objective(args):
    integrator = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    print(integrator.all_probs)
    return -1.0 * integrator.effective_ns_per_day


space = [steps_per_hmc, timestep, extra_chances]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

integrator = inner_objective((best["steps_per_hmc"], best["timestep"], best["extra_chances"]))
best
print(integrator.effective_ns_per_day, integrator.effective_timestep)
