import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "chargedswitchedaccurateljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(system, temperature, timestep, positions, steps=equil_steps, minimize=True)


collision_rate = 1.0 / u.picoseconds
n_steps = 25
Neff_cutoff = 2E4

#integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep)
#integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=25, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.HMCRESPAIntegrator(temperature, steps_per_hmc=25, timestep=timestep * groups[0][1], groups=groups)
#integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=25, timestep=timestep * groups[0][1], groups=groups)
itype = type(integrator).__name__
context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
filename = "./data/%s_%s_%s_%.3f_%d.csv" % (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
print(filename)
integrator.step(equil_steps)
data, start, g, Neff, mu, sigma, stderr = lb_loader.converge(context, n_steps=n_steps, Neff_cutoff=Neff_cutoff, filename=filename)
