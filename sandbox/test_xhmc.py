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

#testsystem = testsystems.LennardJonesFluid()
#testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
#system, positions = testsystem.system, testsystem.positions

#pdb = app.PDBFile("/home/kyleb/src/openmm/openmm/wrappers/python/simtk/openmm/app/data/tip3p.pdb")
#forcefield = app.ForceField('tip3p.xml')

#system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*u.nanometers)
#positions = pdb.positions

system, positions = lb_loader.load_lb()

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(50)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()



collision_rate = 10000.0 / u.picoseconds

timestep = 0.5 * u.femtoseconds
steps_per_hmc = 5
k_max = 4


integrator = hmc_integrators.XHMCIntegrator(temperature, steps_per_hmc, timestep, collision_rate, k_max)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(10)
data = integrator.vstep(50)
data.Enew[data.a == 1].mean(), data.Enew[data.a == 1].std(), data.Enew[data.a == 1].std() / sum(data.a == 1) ** 0.5
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()
#columns = ["a", "flip", "deltaE"] + ["T%d" % i for i in range(4)] + ["pe%d" % i for i in range(4)] + ["ke%d" % i for i in range(4)]
#data[columns]




integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc, timestep, collision_rate)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(10)
data = integrator.vstep(50)
data["a"] = data.accept
data.Enew[data.a == 1].mean(), data.Enew[data.a == 1].std(), data.Enew[data.a == 1].std() / sum(data.a == 1) ** 0.5

