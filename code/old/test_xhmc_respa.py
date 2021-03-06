import lb_loader
import simtk.openmm.app as app
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 3000
temperature = 300. * u.kelvin

hydrogenMass = 3.0 * u.amu

#system, positions = lb_loader.load_lb(hydrogenMass=hydrogenMass)

testsystem = testsystems.DHFRExplicit(hydrogenMass=hydrogenMass, nonbondedCutoff=1.1 * u.nanometers)
system, positions = testsystem.system, testsystem.positions

hmc_integrators.guess_force_groups(system, nonbonded=1, fft=2, others=0)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

%time integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()


collision_rate = 10000.0 / u.picoseconds

groups = [(0, 1), (1, 1), (2, 1)]
timestep = 1.0 * u.femtoseconds
steps_per_hmc = 10
extra_chances = 6




integrator = hmc_integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc, timestep, collision_rate, extra_chances, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(1000)
data = integrator.vstep(25)
integrator.effective_timestep, integrator.effective_ns_per_day


integrator = hmc_integrators.GHMCRESPA(temperature, steps_per_hmc, timestep, collision_rate, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(500)
data = integrator.vstep(25)
integrator.effective_timestep, integrator.effective_ns_per_day



integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc, timestep, collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(500)
data = integrator.vstep(25)
integrator.effective_timestep, integrator.effective_ns_per_day
