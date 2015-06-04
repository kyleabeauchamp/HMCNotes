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

hmc_integrators.guess_force_groups(system, nonbonded=1, others=0, fft=2)


timestep = 2.0 * u.femtoseconds
#integrator = mm.LangevinIntegrator(temperature, 2.0 / u.picoseconds, timestep)
total_steps = 3000
extra_chances = 3
steps_per_hmc = 100
steps = total_steps

steps = total_steps / steps_per_hmc
integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.reset_time()

t0 = time.time()
integrator.step(steps)
dt = time.time() - t0
ns_per_day = (timestep / u.nanoseconds) * total_steps / dt * 60 * 60 * 24
dt, ns_per_day
integrator.ns_per_day
