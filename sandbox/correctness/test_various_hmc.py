import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(system, temperature, timestep, positions, steps=equil_steps, minimize=True)


#collision_rate = 1E-1 / u.picoseconds
collision_rate = 1E0 / u.picoseconds
#collision_rate = 1E1 / u.picoseconds
n_steps = 1  # Number inside integrator.step()
Neff_cutoff = 1E5

# Water box settings
groups = [(0, 2), (1, 1)]
extra_chances = 2
steps_per_hmc = 25
timestep = 2.4208317230875265 * 1.5 * u.femtoseconds

#timestep = timestep * 1.5

integrator = mm.LangevinIntegrator(temperature, collision_rate, 2.0 * u.femtoseconds)
#integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
#integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)
#integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
#integrator = hmc_integrators.HMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)
#integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)
#integrator = hmc_integrators.XCHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, extra_chances=extra_chances)
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, collision_rate=collision_rate)

itype = type(integrator).__name__
context = lb_loader.build(system, integrator, positions, temperature, precision=precision)
filename = "./data/%s_%s_%s_%.3f_%d.csv" % (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
print(filename)
integrator.step(equil_steps)
data, g, Neff, mu, sigma, stderr = lb_loader.converge(context, integrator, n_steps=n_steps, Neff_cutoff=Neff_cutoff, filename=filename)
