import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

platform = mm.Platform_getPlatformByName("CUDA")
platform_properties = dict(CudaPrecision="single")

n_steps = 2500
temperature = 300. * u.kelvin
collision_rate = 1.0 / u.picoseconds

cutoff = 1.0 * u.nanometers
hydrogenMass = 3.0 * u.amu

testsystem = testsystems.DHFRExplicit(hydrogenMass=hydrogenMass, cutoff=cutoff)
system, positions = testsystem.system, testsystem.positions

positions = lb_loader.pre_equil(system, positions, temperature)


steps_per_hmc = 17
timestep = 1.5 * u.femtoseconds
hmc_integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
factor = 1
groups = [(0, 2), (1, 1)]


for i in range(3):
    integrator = hmc_integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
    context = mm.Context(system, integrator, platform, platform_properties)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(1)
    integrator.step(n_steps)
    integrator.vstep(5)
