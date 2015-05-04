from openmmtools import integrators, testsystems
import pymbar
import pandas as pd
import time
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
    

def load_lb(cutoff=1.1 * u.nanometers, constraints=app.HBonds, hydrogenMass=1.0 * u.amu):
    prmtop_filename = "./input/126492-54-4_1000_300.6.prmtop"
    pdb_filename = "./input/126492-54-4_1000_300.6_equil.pdb"

    pdb = app.PDBFile(pdb_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=constraints, hydrogenMass=hydrogenMass)  # Force rigid water here for comparison to other code
    return system, pdb.positions


def set_masses(system):
    total_mass = sum(system.getParticleMass(k) / u.dalton for k in range(system.getNumParticles()))
    particle_mass = (total_mass / float(system.getNumParticles())) * u.dalton
    for i in range(system.getNumParticles()):
        system.setParticleMass(i, particle_mass)


def load_amoeba(hydrogenMass=1.0):
    pdb = app.PDBFile("./sandbox/iamoeba.pdb")
    forcefield = app.ForceField('iamoeba.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*u.nanometers, hydrogenMass=hydrogenMass * u.amu, rigidWater=False)    
    return system, pdb.positions


def converge(context, n_steps=1, Neff_cutoff=1E4, sleep_time=60):
    integrator = context.getIntegrator()
    
    data = None
    
    t00 = time.time()
    t0 = time.time() - t00
    t1 = time.time() - t00
    
    while True:
        integrator.step(n_steps)
        if type(integrator) in ["XHMCIntegrator", "XHMCRESPAIntegrator"]:
            if integrator.getGlobalVariableByName("a") != 1.:
                continue
        
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy() / u.kilojoules_per_mole
        
        current_data = dict(energy=energy)
        
        if data is None:
            data = pd.DataFrame([current_data])
        else:
            data.ix[len(data)] = current_data

        t1 = time.time() - t00
        
        if t1 - t0 < sleep_time:
            continue
        
        t0 = t1
        
        energies = data.energy.values

        [start, g, Neff] = pymbar.timeseries.detectEquilibration(energies, nskip=1000)
        
        mu = energies.mean()
        sigma = energies.std()
        stderr = sigma * Neff ** -0.5
        
        print("t0=%f, energy = %.4f + %.3f, N=%d, start=%d, g=%.4f, Neff=%.4f, stderr=%f" % (t0, mu, sigma, len(energies), start, g, Neff, stderr))

        if Neff > Neff_cutoff:
            return data, start, g, Neff

def build(system, integrator, positions, temperature):
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    return context

def load_lj():
    testsystem = testsystems.LennardJonesFluid()

    system, positions = testsystem.system, testsystem.positions

    positions = np.loadtxt("./sandbox/ljbox.dat")
    length = 2.66723326712
    boxes = ((length, 0, 0), (0, length, 0), (0, 0, length))
    system.setDefaultPeriodicBoxVectors(*boxes)
    
    return testsystem, system, positions

def load(sysname):
    
    temperature = 300. * u.kelvin
    timestep = 2 * u.femtoseconds  # LJ Cluster

    if sysname == "ljbox":
        system, positions = load_lj()[1:]
        integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
    
    if sysname == "water":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]

    if sysname == "density":
        system, positions = load_lb(hydrogenMass=3.0 * u.amu)
        integrators.guess_force_groups(system, nonbonded=1, fft=2, others=0)
        groups = [(0, 2), (1, 1)]
        timestep = 1.0 * u.femtoseconds


    return system, positions, groups, temperature, timestep
