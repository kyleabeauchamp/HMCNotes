from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 1000
platform_name = "CUDA"
precision = "mixed"

sysname = "src"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

max_evals = 500

steps_per_hmc = hp.quniform("steps_per_hmc", 5, 30, 1)
extra_chances = hp.quniform("extra_chances", 1, 4, 1)
timestep = hp.uniform("timestep", 0.25, 1.5)


def inner_objective(args):
    steps_per_hmc, timestep, extra_chances = args
    print("*" * 80)
    print("steps=%d, timestep=%f, extra_chances=%d" % (steps_per_hmc, timestep, extra_chances))
    current_timestep = timestep * u.femtoseconds
    extra_chances = int(extra_chances)
    steps_per_hmc = int(steps_per_hmc)
    integrator = hmc_integrators.XCGHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, extra_chances=extra_chances)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)
    simulation.integrator.step(n_steps)
    return integrator, simulation  # Have to pass simulation to keep it from being garbage collected

def objective(args):
    integrator, simulation = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    print(integrator.all_probs)
    return -1.0 * integrator.effective_ns_per_day


space = [steps_per_hmc, timestep, extra_chances]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

integrator, simulation = inner_objective((best["steps_per_hmc"], best["timestep"], best["extra_chances"]))
best
print(integrator.effective_ns_per_day, integrator.effective_timestep)
