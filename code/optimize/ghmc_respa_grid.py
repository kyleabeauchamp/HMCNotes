import seaborn as sns
sns.set(style="whitegrid")
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
import collections
import itertools
import os
import sklearn.grid_search
pd.set_option('display.width', 1000)

n_iter = 100
n_steps = 500
platform_name = "CUDA"
precision = "mixed"

#sysname = "switchedaccuratebigflexiblewater"
sysname = "switchedaccurateflexiblewater"
filename = "./data/gridsearch/%s.csv" % sysname

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)


grid = sklearn.grid_search.ParameterGrid(dict(
temperature = [temperature],
steps_per_hmc = [25],
group0_iterations = [0, 1, 2],
timestep = np.linspace(0.4, 1.5, 12),
#timestep = np.linspace(0.2, 0.4, 5),
extra_chances = [0],
groups=((0,0,0), (0,1,1), (0,0,1)),
))


def objective(**kwargs):
    groups = kwargs["groups"]
    hmc_integrators.guess_force_groups(system, others=groups[0], nonbonded=groups[1], fft=groups[2])
    integrator = lb_loader.kw_to_int(**kwargs)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name, state=state)
    integrator.reset_time()
    integrator.step(n_steps)
    new_state = simulation.context.getState(getPositions=True, getParameters=True)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    results = dict(
    n_steps=n_steps,
    intname=integrator.__class__.__name__,
    effective_ns_per_day=integrator.effective_ns_per_day,
    acceptance_rate=integrator.acceptance_rate,
    effective_timestep=integrator.effective_timestep / u.femtoseconds,
    time_per_step=integrator.time_per_step,
    )
    print(results)
    return results, new_state


COLUMNS = ['RESPA', 'acceptance_rate', 'effective_ns_per_day', 'effective_timestep', 'extra_chances', 'group0_iterations', 'groups', 'intname', 'n_steps', 'steps_per_hmc', 'temperature', 'time_per_step', 'timestep']

for i in range(n_iter):
    for parameters in grid:
        print(i)
        print(parameters)
        results_i, state = objective(**parameters)
        results_i.update(parameters)
        results_i["RESPA"] = str(results_i["group0_iterations"]) + "x" + str(results_i["groups"])
        results = pd.DataFrame([results_i])
        results = results[COLUMNS]
        header = True if not os.path.exists(filename) else False
        results.to_csv(filename, mode='a', header=header)
