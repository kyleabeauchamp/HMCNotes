import time
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"

sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, use_hmc=False)


del simulation, integrator
timestep = 4.25 * u.femtoseconds

total_steps = 50000
extra_chances = 10
steps_per_hmc = 100
collision_rate = 3E0 / u.picoseconds

steps = total_steps / steps_per_hmc
integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.reset_time()
integrator.step(steps)

c = integrator.all_counts
p = integrator.all_probs
c
p
integrator.effective_timestep, integrator.effective_ns_per_day
integrator.ns_per_day, integrator.time_per_step
sqrt(1 - integrator.b)
