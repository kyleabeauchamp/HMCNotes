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

for k, integrator in enumerate(lb_loader.get_grid(temperature, timestep, langevin_timestep, groups)):
    itype = type(integrator).__name__
    simulation = lb_loader.build(testsystem, integrator, temperature)
    simulation.step(5)

    csv_filename = "./data/%s_int%d.csv" % (sysname, k)
    output_frequency = 100 if "Langevin" in itype else 5
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, temperature=True, density=True, elapsedTime=True))
    simulation.runForClockTime(1.0 * u.minutes)
