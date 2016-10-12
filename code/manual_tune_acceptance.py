import time
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"
platform_name = "CUDA"

#sysname = "switchedaccurateflexiblewater"
sysname = "switchedljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

timestep = 0.1 * u.femtoseconds

extra_chances = 3
steps_per_hmc = 1
#collision_rate = 3E0 / u.picoseconds
collision_rate = None

steps = 1
#groups = [(0, 1), (1, 1)]
groups = [(0, 1)]
integrator = hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
#integrator = mm.MTSIntegrator(dt=timestep, groups=groups)
f = lambda x: integrator.getGlobalVariableByName(x)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)

#integrator.reset_time()

print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

integrator.step(steps)

print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

#print(integrator.all_counts)
#print(integrator.all_probs)
#print(integrator.effective_timestep, integrator.effective_ns_per_day)
#print(integrator.ns_per_day, integrator.time_per_step)


integrator.getGlobalVariableByName("Eold")
integrator.getGlobalVariableByName("Enew")
for i in range(integrator.getNumComputations()): print(integrator.getComputationStep(i))

print(integrator.acceptance_rate)
