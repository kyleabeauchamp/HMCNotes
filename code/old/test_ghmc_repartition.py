import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin

for hydrogenMass in [1.0, 2.0, 3.0, 3.5]:
    hydrogenMass = hydrogenMass * u.amu
    #system, positions = lb_loader.load_lb(hydrogenMass=hydrogenMass)
    testsystem = testsystems.DHFRExplicit(hydrogenMass=hydrogenMass)
    system, positions = testsystem.system, testsystem.positions

    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(10000)
    positions = context.getState(getPositions=True).getPositions()

    collision_rate = 1.0 / u.picoseconds
    timestep = 1.5 * u.femtoseconds
    steps_per_hmc = 12
    k_max = 3

    integrator = hmc_integrators.GHMC2(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    integrator.step(2500)
    data = integrator.vstep(5)
