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
    
    t0 = time.time()
    t1 = time.time()
    
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

        if time.time() - t0 < sleep_time:
            continue

        energies = data.energy.values

        [t0, g, Neff] = pymbar.timeseries.detectEquilibration(energies, nskip=1000)
        print("energy = %.4f + %.3f, N=%d, g=%.4f, Neff=%.4f" % (energies.mean(), energies.std(), len(energies), g, Neff))

        if Neff > Neff_cutoff:
            return data, t0, g, Neff
