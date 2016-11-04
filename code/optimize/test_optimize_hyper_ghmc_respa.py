from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 250
platform_name = "CUDA"
precision = "mixed"

sysname = "switchedaccuratebigflexiblewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

max_evals = 50

steps_per_hmc = hp.quniform("steps_per_hmc", 25, 35, 1)
group0_iterations = hp.quniform("group0_iterations", 1, 3, 1)
timestep = hp.uniform("timestep", 0.2, 0.5)
#num_to_groups = lambda group0_iterations: ((0, 1), (1, group0_iterations))
num_to_groups = lambda group0_iterations: ((0, group0_iterations), (1, 1))


def inner_objective(args):
    steps_per_hmc, timestep, group0_iterations = args
    print("steps=%d, timestep=%f, extra_chances=%d, grp = %d" % (steps_per_hmc, timestep, 0, group0_iterations))
    current_timestep = timestep * u.femtoseconds
    steps_per_hmc = int(steps_per_hmc)
    group0_iterations = int(group0_iterations)
    groups = num_to_groups(group0_iterations)
    integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, groups=groups)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)
    integrator.reset_time()
    integrator.step(n_steps)
    return integrator, simulation  # Have to pass simulation to keep it from being garbage collected

def objective(args):
    print("*" * 80)
    integrator, simulation = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    return -1.0 * integrator.effective_ns_per_day


space = [steps_per_hmc, timestep, group0_iterations]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

integrator, simulation = inner_objective((best["steps_per_hmc"], best["timestep"], best["group0_iterations"]))
best
print(integrator.effective_ns_per_day, integrator.effective_timestep)
