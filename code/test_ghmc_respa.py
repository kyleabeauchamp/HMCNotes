import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

platform = mm.Platform_getPlatformByName("CUDA")
platform_properties = dict(CudaPrecision="single")

n_steps = 500
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds

timestep = 1.0 * u.femtoseconds
#timestep = 2.16 * u.femtoseconds
steps_per_hmc = 18

#cutoff = 1.0 * u.nanometers

system, positions = lb_loader.load_lb()
#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=cutoff)  # Around 1060 molecules of water
#system, positions = testsystem.system, testsystem.positions

integrators.guess_force_groups(system, nonbonded=1, fft=1)

positions = lb_loader.pre_equil(system, positions, temperature)

factor = 3
groups = [(0, factor), (1, 1)]

integrator = integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator, platform, platform_properties)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
integrator.step(n_steps)
print(pd.Series(integrator.summary))


integrator = integrators.GHMC2(temperature, steps_per_hmc, timestep, collision_rate)
context = mm.Context(system, integrator, platform, platform_properties)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
integrator.step(n_steps)
print(pd.Series(integrator.summary))


import simtk.openmm.app as app
integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 1.0 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

import time
integrator.step(1)
t0 = time.time()
integrator.step(1000)
dt = time.time() - t0
ns_per_second = 1E-3 / dt
ns_per_day = 60 * 60 * 24 * ns_per_second
ns_per_day




integrator = integrators.GHMC2(temperature, 10, 1.0 * u.femtoseconds, collision_rate)
context = mm.Context(system, integrator, platform, platform_properties)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
t0 = time.time()
integrator.step(100)
dt = time.time() - t0
ns_per_second = 1E-3 / dt
ns_per_day = 60 * 60 * 24 * ns_per_second
ns_per_day

integrator.effective_ns_per_day
