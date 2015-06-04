import time
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"

sysname = "dhfr"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, use_hmc=False)


del simulation, integrator
timestep = 2.5 * u.femtoseconds

total_steps = 3000
extra_chances = 3
steps_per_hmc = 170

#groups = [(0, 1), (1, 2), (2, 4)]
#hmc_integrators.guess_force_groups(system, nonbonded=1, others=2, fft=0)

groups = [(0, 1), (1, 2)]
hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)

#groups = [(0, 1)]
#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)

#steps = total_steps
#integrator = mm.MTSIntegrator(timestep, groups)

steps = total_steps / steps_per_hmc
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)
integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups)
#integrator = hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.reset_time()

t0 = time.time()
integrator.step(steps)
dt = time.time() - t0
ns_per_day = (timestep / u.nanoseconds) * total_steps / dt * 60 * 60 * 24

c = integrator.all_counts
p = integrator.all_probs
c
p
integrator.effective_timestep, integrator.effective_ns_per_day
integrator.ns_per_day
dt, ns_per_day
