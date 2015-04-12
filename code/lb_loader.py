import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems

def load_lb():
    prmtop_filename = "/home/kyleb/liquid_benchmark_3_14/equil/14314-42-2_1000_303.2.prmtop"
    pdb_filename = "/home/kyleb/liquid_benchmark_3_14/equil/14314-42-2_1000_303.2_equil.pdb"

    pdb = app.PDBFile(pdb_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HAngles)  # Force rigid water here for comparison to other code
    return system, pdb.positions
