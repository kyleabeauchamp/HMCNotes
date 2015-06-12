import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True)

collision_rate = 0.1 / u.picoseconds

del simulation, integrator

timestep = 4.25 * u.femtoseconds
extra_chances = 10
steps_per_hmc = 619
output_frequency = 1

groups = [(0, 1), (1, 2)]
hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=fixed(collision_rate))

integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)

itype = type(integrator).__name__

simulation = lb_loader.build(testsystem, integrator, temperature)
integrator.reset_time()
simulation.step(100)
print(timestep, integrator.acceptance_rate, integrator.effective_timestep)

prms = ( sysname, itype, timestep / u.femtoseconds, lb_loader.fixunits(collision_rate))
csv_filename = "./data/%s_%s_%.3f_%d.csv" % prms
pdb_filename = "./data/%s_%s_%.3f_%d.pdb" % prms
dcd_filename = "./data/%s_%s_%.3f_%d.dcd" % prms

simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, temperature=True, density=True, elapsedTime=True))
simulation.reporters.append(app.PDBReporter(pdb_filename, nsteps - 1))
simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))

simulation.runForClockTime(30 * u.minutes)
