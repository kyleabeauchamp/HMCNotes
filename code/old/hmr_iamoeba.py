import lb_loader
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

pdb = app.PDBFile("./sandbox/iamoeba.pdb")
forcefield = app.ForceField('iamoeba.xml')

n_steps = 4000
temperature = 300. * u.kelvin

masses = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
data = []
for hydrogenMass in masses:
    print(hydrogenMass)
    positions = pdb.positions
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*u.nanometers, hydrogenMass=hydrogenMass * u.amu, rigidWater=False)
    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
    
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(100)
    positions = context.getState(getPositions=True).getPositions()

    collision_rate = 1.0 / u.picoseconds
    timestep = 0.5 * u.femtoseconds
    steps_per_hmc = 12

    integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    integrator.step(n_steps)


    integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    integrator.step(1)
    integrator.step(n_steps)
    data.append(integrator.summary)
    data[-1]["mass"] = hydrogenMass

data = pd.DataFrame(data)
data.to_csv("./tables/iamoeba_raw_performance.csv")
data

