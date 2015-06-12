from hyperopt import fmin, tpe, hp
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 100
precision = "mixed"

sysname = "alanineexplicit"

system, positions, groups, temperature, timestep0, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep0, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)

max_evals = 100

hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
steps_per_hmc = hp.quniform("steps_per_hmc", 600, 625, 1)
extra_chances = hp.quniform("extra_chances", 4, 10, 1)
timestep = hp.uniform("timestep", 2.75, 4.5)
groups = [(0, 1), (1, 2)]

def inner_objective(args):
    steps_per_hmc, timestep, extra_chances = args
    print("steps=%d, timestep=%f, extra_chances=%d" % (steps_per_hmc, timestep, extra_chances))
    current_timestep = timestep * u.femtoseconds
    extra_chances = int(extra_chances)
    steps_per_hmc = int(steps_per_hmc)
    integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, extra_chances=extra_chances, groups=groups)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)
    integrator.reset_time()
    integrator.step(n_steps)
    return simulation, integrator

def objective(args):
    simulation, integrator = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    print(integrator.all_probs)
    return -1.0 * integrator.effective_ns_per_day


space = [steps_per_hmc, timestep, extra_chances]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

simulation, integrator = inner_objective((best["steps_per_hmc"], best["timestep"], best["extra_chances"]))
best
print(integrator.effective_ns_per_day, integrator.effective_timestep)
