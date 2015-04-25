import lb_loader
import simtk.openmm.app as app
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin

hydrogenMass = 3.0 * u.amu

system, positions = lb_loader.load_amoeba()
integrators.guess_force_groups(system, multipole=2)

#system, positions = lb_loader.load_lb(hydrogenMass=hydrogenMass)
#integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1000)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()



collision_rate = 10000.0 / u.picoseconds

groups = [(0, 4), (1, 2), (2, 1)]
timestep = 1.0 * u.femtoseconds
steps_per_hmc = 10
k_max = 6


integrator = integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc, timestep, collision_rate, k_max, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(500)
data = integrator.vstep(25)


integrator = integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(500)
data = integrator.vstep(25)




integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc, timestep, collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(500)
data = integrator.vstep(25)
