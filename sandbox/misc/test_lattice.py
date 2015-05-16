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




n = 7 ** 3
r = 1.0

xyz, box = build_lattice(n, r)

traj = generate_dummy_trajectory(xyz, box)
traj.save("./out.pdb")

testsystem = LennardJonesFluid(nparticles=1000, hcp=True)

system, positions = testsystem.system, testsystem.positions
temperature = 25*u.kelvin

integrator = hmc_integrators.HMCIntegrator(temperature, steps_per_hmc=25, timestep=1.0*u.femtoseconds)
context = lb_loader.build(system, integrator, positions, temperature)
context.getState(getEnergies=True).getPotentialEnergy()
mm.LocalEnergyMinimizer.minimize(context)
context.getState(getEnergies=True).getPotentialEnergy()
integrator.step(400)
context.getState(getEnergies=True).getPotentialEnergy()
