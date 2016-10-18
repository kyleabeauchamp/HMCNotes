from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 400
platform_name = "CUDA"
precision = "mixed"

sysname = "switchedaccurateflexiblewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

max_evals = 25

steps_per_hmc = hp.quniform("steps_per_hmc", 15, 30, 1)
timestep = hp.uniform("timestep", 0.9, 1.2)

def inner_objective(args):
    steps_per_hmc, timestep = args
    print("*" * 80)
    print("steps=%d, timestep=%f, extra_chances=%d" % (steps_per_hmc, timestep, 0))
    current_timestep = timestep * u.femtoseconds
    steps_per_hmc = int(steps_per_hmc)
    integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)
    simulation.integrator.step(n_steps)
    return integrator, simulation  # Have to pass simulation to keep it from being garbage collected

def objective(args):
    integrator, simulation = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    return -1.0 * integrator.effective_ns_per_day


space = [steps_per_hmc, timestep]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, verbose=1)

integrator, simulation = inner_objective((best["steps_per_hmc"], best["timestep"]))
print(best)
print(integrator.effective_ns_per_day, integrator.effective_timestep)
