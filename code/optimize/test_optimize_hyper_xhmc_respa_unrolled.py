from hyperopt import fmin, tpe, hp
import lb_loader
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

n_steps = 200
precision = "mixed"

sysname = "switchedaccuratenptwater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)


#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=1)

max_evals = 10

steps_per_hmc = hp.quniform("steps_per_hmc", 50, 55, 1)
extra_chances = hp.quniform("extra_chances", 6, 10, 1)
outer_timestep = hp.uniform("outer_timestep", 5.0, 6.0)
respa_factor0 = hp.quniform("respa_factor0", 2, 3, 1)
#respa_factor1 = hp.quniform("respa_factor1", 1, 2, 1)


def inner_objective(args):
    steps_per_hmc, outer_timestep, extra_chances, respa_factor0 = args
    print("steps=%d, outer_timestep=%f, extra_chances=%d, respa_factor0=%d" % (steps_per_hmc, outer_timestep, extra_chances, respa_factor0))    
    #steps_per_hmc, outer_timestep, extra_chances, respa_factor0, respa_factor1 = args
    #print("steps=%d, outer_timestep=%f, extra_chances=%d, respa_factor0=%d respa_factor1=%d" % (steps_per_hmc, outer_timestep, extra_chances, respa_factor0, respa_factor1))
    current_timestep = outer_timestep * u.femtoseconds
    extra_chances = int(extra_chances)
    steps_per_hmc = int(steps_per_hmc)
    respa_factor0 = int(respa_factor0)
    #respa_factor1 = int(respa_factor1)
    #groups = [(0, respa_factor0), (1, respa_factor1), (2, 1)]
    groups = [(0, respa_factor0), (1, 1)]
    integrator = hmc_integrators.UnrolledXCHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=current_timestep, extra_chances=extra_chances, groups=groups)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)
    integrator.step(n_steps)
    return simulation, integrator

def objective(args):
    simulation, integrator = inner_objective(args)
    print("eff_ns_per_day=%f, eff_dt=%f" % (integrator.effective_ns_per_day, integrator.effective_timestep / u.femtoseconds))
    print(integrator.all_probs)
    return -1.0 * integrator.effective_ns_per_day


#space = [steps_per_hmc, outer_timestep, extra_chances, respa_factor0, respa_factor1]
space = [steps_per_hmc, outer_timestep, extra_chances, respa_factor0]
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)

#simulation, integrator = inner_objective((best["steps_per_hmc"], best["outer_timestep"], best["extra_chances"], best["respa_factor0"], best["respa_factor1"]))
simulation, integrator = inner_objective((best["steps_per_hmc"], best["outer_timestep"], best["extra_chances"], best["respa_factor0"]))
best
print(integrator.effective_ns_per_day, integrator.effective_timestep)
