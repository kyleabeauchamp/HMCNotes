import numpy as np
import simtk.openmm as mm
from simtk import unit as u
import simtk.openmm.app as app

temperature = 300. * u.kelvin

pdb = app.PDBFile("/home/kyleb/src/openmm/openmm/wrappers/python/simtk/openmm/app/data/tip3p.pdb")
forcefield = app.ForceField('tip3p.xml')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*u.nanometers)
positions = pdb.positions


integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)

mm.LocalEnergyMinimizer_minimize(context, 0.1)
positions = context.getState(getPositions=True).getPositions()
context.setVelocitiesToTemperature(temperature)


integrator.step(500)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()


integrator = mm.CustomIntegrator(0.0005)
integrator.addGlobalVariable("U", 0.0)
integrator.addGlobalVariable("K", 0.0)
integrator.addPerDofVariable("x1", 0)
integrator.addUpdateContextState()
integrator.addComputePerDof("v", "v+0.5*dt*f/m")
integrator.addComputePerDof("x", "x+dt*v")
integrator.addComputePerDof("x1", "x")
integrator.addConstrainPositions()
integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
integrator.addConstrainVelocities()
integrator.addComputeGlobal("U", "energy")
integrator.addComputeSum("K", "0.5*m*v*v")

context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)


integrator.step(1)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()

integrator.getGlobalVariableByName("U")
integrator.getGlobalVariableByName("K")
