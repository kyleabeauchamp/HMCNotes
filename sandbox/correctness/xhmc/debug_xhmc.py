import pymbar
import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

sysname = "customsplitho"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(system, temperature, timestep, positions, steps=equil_steps, minimize=True, steps_per_hmc=steps_per_hmc)
E0 = (3/2.) * testsystem.n_particles * testsystems.kB * temperature / u.kilojoules_per_mole

integrator = hmc_integrators.XCHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.step(3000)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.acceptance_rate
positions = context.getState(getPositions=True).getPositions()
output = integrator.vstep(25)

data = []
for i in range(100000):
    integrator.step(1)
    energy = context.getState(getEnergy=True).getPotentialEnergy() / u.kilojoules_per_mole
    data.append(dict(accept=integrator.accept, energy=energy))

data = pd.DataFrame(data)
energies = data.energy.values

energies.mean()
energies.mean() - E0

g = pymbar.timeseries.statisticalInefficiency(energies)
Neff = len(energies) / g
stderr = energies.std() / (Neff ** 0.5)

data, g, Neff, mu, sigma, stderr = lb_loader.converge(context, n_steps=1, Neff_cutoff=1E4)

data = [dict(energy=E0)]
data = pd.DataFrame(data)
while True:
    integrator.step(1)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() / u.kilojoules_per_mole
    current_data = dict(energy=energy)
    data.ix[len(data)] = current_data
