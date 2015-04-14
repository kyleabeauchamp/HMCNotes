from simtk.openmm import app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u

def pre_equil(system, positions, temperature):
    integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.5 * u.femtoseconds)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(5000)
    positions = context.getState(getPositions=True).getPositions()
    return positions
    

def load_lb(cutoff=1.1 * u.nanometers):
    prmtop_filename = "./input/126492-54-4_1000_300.6.prmtop"
    pdb_filename = "./input/126492-54-4_1000_300.6_equil.pdb"

    pdb = app.PDBFile(pdb_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HBonds)  # Force rigid water here for comparison to other code
    return system, pdb.positions
