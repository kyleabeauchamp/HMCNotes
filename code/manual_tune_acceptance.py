import time
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"

sysname = "switchedaccurateflexiblewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False)

timestep = 1.0 * u.femtoseconds

extra_chances = 3
steps_per_hmc = 100
#collision_rate = 3E0 / u.picoseconds
collision_rate = None

steps = 200
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
import pickle
integrator = pickle.load(open('./test.pkl'))
f = lambda x: integrator.getGlobalVariableByName(x)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.reset_time()
integrator.step(steps)

print(integrator.all_counts)
print(integrator.all_probs)
print(integrator.effective_timestep, integrator.effective_ns_per_day)
print(integrator.ns_per_day, integrator.time_per_step)
