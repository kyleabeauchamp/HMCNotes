import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

precision = "mixed"

#sysname = "switchedaccuratewater"
sysname = "alanine"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True)

"""
#nsteps = 20000000
nsteps = 10000000
output_frequency = 250
collision_rate = 100. / u.picoseconds
timestep = 2.0 * u.femtoseconds
integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
"""


nsteps = 500000
collision_rate = 1. / u.picoseconds  # Not used but necessary for filename convention.
output_frequency = 5
timestep = 4.029 * u.femtoseconds
extra_chances = 6
steps_per_hmc = 32
integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)


"""
nsteps = 500000
collision_rate = 100. / u.picoseconds
output_frequency = 5
timestep = 3.988 * u.femtoseconds
extra_chances = 4
steps_per_hmc = 33
integrator = hmc_integrators.XCGHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
"""

"""
nsteps = 100000
output_frequency = 11
timestep = 2.8861832226977824 * u.femtoseconds
extra_chances = 1
steps_per_hmc = 23
groups = [(0, 2), (1, 2), (2, 1)]
"""


"""
nsteps = 100000
output_frequency = 11
extra_chances = 2
steps_per_hmc = 25
timestep = 3.63125 * u.femtoseconds
groups = [(0, 2), (1, 1)]
#integrator = hmc_integrators.XCHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, extra_chances=extra_chances)
collision_rate = 100.0 / u.picoseconds
integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, extra_chances=extra_chances, collision_rate=collision_rate)
"""


itype = type(integrator).__name__

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)
simulation.step(100)
print(timestep, integrator.acceptance_rate, integrator.effective_timestep)

prms = (precision, sysname, itype, timestep / u.femtoseconds, collision_rate * u.picoseconds)
csv_filename = "./data/%s_%s_%s_%.3f_%d.csv" % prms
pdb_filename = "./data/%s_%s_%s_%.3f_%d.pdb" % prms
dcd_filename = "./data/%s_%s_%s_%.3f_%d.dcd" % prms



simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, temperature=True, density=True, elapsedTime=True))
simulation.reporters.append(app.PDBReporter(pdb_filename, nsteps - 1))
simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
simulation.step(nsteps)
del simulation, integrator
