import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

sysname = "chargedswitchedaccurateljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True)

collision_rate = None

#del simulation, integrator

timestep = 40. * u.femtoseconds
extra_chances = 2
steps_per_hmc = 50
output_frequency = 1

integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)

itype = type(integrator).__name__

simulation = lb_loader.build(testsystem, integrator, temperature)

for i in range(1):
    simulation.step(100)
    print("i=%d" % i)
    print("Counts")
    print(integrator.all_counts)
    print(integrator.getGlobalVariableByName("nflip"))
    print(integrator.getGlobalVariableByName("terminal_chance"))
