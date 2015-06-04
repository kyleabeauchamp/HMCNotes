import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "dhfr"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True, use_hmc=False)


hmc_integrators.guess_force_groups(system, nonbonded=1, others=0, fft=2)
timestep = 4.0 * u.femtoseconds
steps_per_hmc = 100
extra_chances = 6
groups = [(0, 4), (1, 2), (2, 1)]
integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups)

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)
integrator.step(50)


out = integrator.vstep(10)
c = integrator.all_counts
p = integrator.all_probs
c
p
integrator.effective_timestep, integrator.effective_ns_per_day
testsystem.positions = simulation.context.getState(getPositions=True).getPositions()

del simulation, integrator
integrator = mm.LangevinIntegrator(temperature, 2.0 / u.picoseconds, 2.0 * u.femtoseconds)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

import time
t0 = time.time()
integrator.step(500)
t1 = time.time()
dt = t1 - t0
