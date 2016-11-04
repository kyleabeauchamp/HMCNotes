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
import sklearn.grid_search
pd.set_option('display.width', 1000)

n_steps = 3000
platform_name = "CUDA"
precision = "mixed"

sysname = "switchedaccuratebigflexiblewater"
filename = "./data/gridsearch/%s.csv" % sysname

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

num_to_groups = lambda group0_iterations: ((0, group0_iterations), (1, 1))

grid = sklearn.grid_search.ParameterGrid(dict(
steps_per_hmc = [25],
group0_iterations = [0, 1, 2],
#timestep = np.linspace(0.2, 1.0, 10),
timestep = np.linspace(0.1, 0.3, 12),
extra_chances = [0, 1, 2],
))

def kw_to_int(**kwargs):
    if group0_iterations > 0:
        integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, groups=groups)
    else:
        integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep)

    return integrator

def objective(**kwargs):
    steps_per_hmc = kwargs["steps_per_hmc"]
    timestep = kwargs["timestep"]
    group0_iterations = kwargs["group0_iterations"]
    print("*" * 80)
    print("steps=%d, timestep=%f, extra_chances=%d, grp = %d" % (steps_per_hmc, timestep, 0, group0_iterations))
    current_timestep = timestep * u.femtoseconds
    steps_per_hmc = int(steps_per_hmc)
    group0_iterations = int(group0_iterations)
    groups = num_to_groups(group0_iterations)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)
    integrator.reset_time()
    integrator.step(n_steps)
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
    return results

try:
    results = pd.read_csv(filename, index_col=0)
    results = list(results.T.to_dict().values())
except IOError:
    results = []

for parameters in grid:
    print(parameters)
    results_i = objective(**parameters)
    results_i.update(parameters)
    results.append(results_i)

results = pd.DataFrame(results)

results.to_csv(filename)

#sns.factorplot(x="timestep", y="effective_ns_per_day", hue="group0_iterations", data=results)
sns.lmplot(x="timestep", y="effective_ns_per_day", hue="group0_iterations", data=results, ci=None, order=2)
