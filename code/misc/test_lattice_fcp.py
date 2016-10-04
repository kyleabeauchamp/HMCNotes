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

def build_lattice_cell_old(r=1.0):
    #r = 1.0  # Sphere size
    L = r * 2.0 * (1.0 + 2 ** 0.5)  # Box edge length
    C = L / 2.  # Half the edge length, e.g. the center

    # First make planes with zero z offset
    #plane = np.array([[r, r, 0], [L - r, r, 0], [C, C, 0], [r, L - r, 0], [L - r, L - r, 0]])
    #midplane = np.array([[C, r, 0], [r, C, 0], [L - r, C, 0], [C, L - r, 0]])
    z0 = r
    z1 = C
    z2 = L - r
    xyz = np.concatenate((plane + z0, midplane + z1, plane + z2))
    return xyz, L

def build_lattice_cell():
    #xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0 ,0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    #xyz.extend([[0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [1, 0.5, 0.5], [0.5, 0.5, 1], [0.5, 1, 1]])
    xyz = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]
    xyz = np.array(xyz)

    return xyz



def build_lattice(n_particles):
    n = ((n_particles / 4.) ** (1 / 3.))

    if np.abs(n - np.round(n)) > 1E-10:
        raise(ValueError("Must input 14 n^3 particles for some integer n!"))
    else:
        n = int(np.round(n))

    xyz = []
    cell = build_lattice_cell()
    x, y, z = np.eye(3)
    for atom, (i, j, k) in enumerate(itertools.product(np.arange(n), repeat=3)):
        xi = cell + i * x + j * y + k * z
        xyz.append(xi)

    xyz = np.concatenate(xyz)

    return xyz, n



n = 4 * (2 ** 3)

xyz, box = build_lattice(n)

traj = generate_dummy_trajectory(xyz, box)
traj.save("./out.pdb")
len(xyz)

x = np.linspace(0, 16, 5000)
y = np.array([f(xi) for xi in x])
plot(x, y)


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
