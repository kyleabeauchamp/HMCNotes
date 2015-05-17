import lb_loader
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
import pandas as pd
import mdtraj as md
import pandas as pdb
import spack
import itertools
import numpy as np
from openmmtools.testsystems import build_lattice, generate_dummy_trajectory, LennardJonesFluid


nparticles = 4 * (4 ** 3)

xyz, box = build_lattice(nparticles)

traj = generate_dummy_trajectory(xyz, box)
traj.save("./out.pdb")
len(xyz)


testsystem = LennardJonesFluid(nparticles=nparticles, lattice=True)

system, positions = testsystem.system, testsystem.positions
temperature = 25*u.kelvin

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=1.0*u.femtoseconds)
context = lb_loader.build(system, integrator, positions, temperature)

state = context.getState(getPositions=True, getParameters=True)
xyz = state.getPositions(asNumpy=True) / u.nanometer
state.getPeriodicBoxVectors()


context.getState(getEnergy=True).getPotentialEnergy()
mm.LocalEnergyMinimizer.minimize(context)
context.getState(getEnergy=True).getPotentialEnergy()
integrator.step(1)
context.getState(getEnergy=True).getPotentialEnergy()

integrator.step(100)
context.getState(getEnergy=True).getPotentialEnergy()
