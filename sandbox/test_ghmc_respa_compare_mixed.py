import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

platform = mm.Platform_getPlatformByName("CUDA")


n_steps = 1500
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds

cutoff = 1.0 * u.nanometers
hydrogenMass = 1.0 * u.amu

testsystem = testsystems.DHFRExplicit(hydrogenMass=hydrogenMass, cutoff=cutoff)
system, positions = testsystem.system, testsystem.positions

positions = lb_loader.pre_equil(system, positions, temperature)



platform_properties = dict(CudaPrecision="mixed")
steps_per_hmc = 17
timestep = 1.5 * u.femtoseconds
integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
factor = 1
groups = [(0, 2), (1, 1)]

integrator = integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator, platform, platform_properties)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
integrator.step(n_steps)
integrator.vstep(5)



platform_properties = dict(CudaPrecision="single")
steps_per_hmc = 17
timestep = 1.5 * u.femtoseconds
integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
factor = 1
groups = [(0, 2), (1, 1)]

integrator = integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator, platform, platform_properties)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1)
integrator.step(n_steps)
integrator.vstep(5)
