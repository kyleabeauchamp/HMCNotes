from simtk.openmm import app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u

def load_lb(cutoff=1.1 * u.nanometers):
    prmtop_filename = "./input/126492-54-4_1000_300.6.prmtop"
    pdb_filename = "./input/126492-54-4_1000_300.6_equil.pdb"

    pdb = app.PDBFile(pdb_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HAngles)  # Force rigid water here for comparison to other code
    return system, pdb.positions
