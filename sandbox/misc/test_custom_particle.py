import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

testsystem = testsystems.CustomPotentialTestSystem()
system, positions = testsystem.system, testsystem.positions

temperature = 300 * u.kelvin
timestep = 50.0 * u.femtoseconds

integrator = integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(50)
output = integrator.vstep(10)

data, start, g, Neff, mu, sigma, stderr = lb_loader.converge(context, Neff_cutoff=1E5)

E0 = (3/2.) * testsystem.n_particles * testsystems.kB * temperature / u.kilojoules_per_mole
