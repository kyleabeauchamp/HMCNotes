from openmmtools import hmc_integrators, testsystems
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


def converge(context, n_steps=1, Neff_cutoff=1E4, sleep_time=45, filename=None):
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

        energies = data.energy.values
        #[start, g, Neff] = pymbar.timeseries.detectEquilibration(energies, nskip=1000)  # THIS IS TOO SLOW FOR ITERATIVE USE!
        start = 0  # Requires some pre-equilibration
        g = pymbar.timeseries.statisticalInefficiency(energies)
        Neff = len(energies) / g

        mu = energies[start:].mean()
        sigma = energies[start:].std()
        stderr = sigma * Neff ** -0.5

        print("t0=%f, energy = %.4f + %.3f, N=%d, start=%d, g=%.4f, Neff=%.4f, stderr=%f" % (t0, mu, sigma, len(energies), start, g, Neff, stderr))

        if filename is not None:
            data.to_csv(filename)

        t0 = t1

        if Neff > Neff_cutoff:
            return data, start, g, Neff, mu, sigma, stderr

def build(system, integrator, positions, temperature, precision="mixed"):

    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': precision}
        context = mm.Context(system, integrator, platform, properties)
    except:
        print("Warning: no CUDA platform found, not specifying precision.")
        context = mm.Context(system, integrator)

    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    return context

def load_lj(cutoff=None, dispersion_correction=False, switch_width=None, shift=False):
    reduced_density = 0.3
    testsystem = testsystems.LennardJonesFluid(nparticles=2000, reduced_density=reduced_density, dispersion_correction=dispersion_correction, cutoff=cutoff, switch_width=switch_width, shift=shift)

    system, positions = testsystem.system, testsystem.positions

    #positions = np.loadtxt("./sandbox/ljbox.dat")
    #length = 2.66723326712
    #boxes = ((length, 0, 0), (0, length, 0), (0, 0, length))
    #system.setDefaultPeriodicBoxVectors(*boxes)

    return testsystem, system, positions

def load(sysname):
    cutoff = 0.9 * u.nanometers
    temperature = 300. * u.kelvin
    langevin_timestep = 0.5 * u.femtoseconds

    if sysname == "ljbox":
        testsystem, system, positions = load_lj()
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds

    if sysname == "longljbox":
        testsystem, system, positions = load_lj(cutoff=1.333*u.nanometers)
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds

    if sysname == "shortljbox":
        testsystem, system, positions = load_lj(cutoff=0.90*u.nanometers)
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds

    if sysname == "shiftedljbox":
        testsystem, system, positions = load_lj(shift=True)
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds


    if sysname == "switchedljbox":
        testsystem, system, positions = load_lj(switch_width=0.34*u.nanometers)
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds


    if sysname == "switchedshortljbox":
        testsystem, system, positions = load_lj(cutoff=0.90*u.nanometers, switch_width=0.34*u.nanometers)
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 16 * u.femtoseconds
        langevin_timestep = 2 * u.femtoseconds

    if sysname == "cluster":
        testsystem = testsystems.LennardJonesCluster(nx=8, ny=8, nz=8)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 40 * u.femtoseconds

    if sysname == "shortbigcluster":
        testsystem = testsystems.LennardJonesCluster(nx=20, ny=20, nz=20, cutoff=0.75*u.nanometers)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 10 * u.femtoseconds

    if sysname == "switchedshortbigcluster":
        testsystem = testsystems.LennardJonesCluster(nx=20, ny=20, nz=20, cutoff=0.75*u.nanometers, switch_width=0.1*u.nanometers)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 10 * u.femtoseconds

    if sysname == "bigcluster":
        testsystem = testsystems.LennardJonesCluster(nx=20, ny=20, nz=20, cutoff=1.25*u.nanometers)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 10 * u.femtoseconds

    if sysname == "shortcluster":
        testsystem = testsystems.LennardJonesCluster(nx=8, ny=8, nz=8, cutoff=0.75*u.nanometers)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0)
        groups = [(0, 1)]
        temperature = 25. * u.kelvin
        timestep = 40 * u.femtoseconds


    if sysname == "water":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=None)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds

    if sysname == "switchedwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds

    if sysname == "longswitchedwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.5*u.nanometers, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds


    if sysname == "rfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=None)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds

    if sysname == "switchedrfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds

    if sysname == "longswitchedrfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.5*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        timestep = 1.5 * u.femtoseconds


    if sysname == "density":
        system, positions = load_lb(hydrogenMass=3.0 * u.amu)
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
        groups = [(0, 4), (1, 1)]
        timestep = 2.0 * u.femtoseconds

    if sysname == "dhfr":
        testsystem = testsystems.DHFRExplicit(nonbondedCutoff=cutoff, nonbondedMethod=app.PME)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
        groups = [(0, 4), (1, 1)]
        timestep = 1.0 * u.femtoseconds

    if sysname == "ho":
        K = 90.0 * u.kilocalories_per_mole / u.angstroms**2
        mass = 39.948 * u.amu
        timestep = np.sqrt(mass / K) * 0.4
        testsystem = testsystems.HarmonicOscillatorArray()
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 1)]

    if sysname == "customho":
        timestep = 50.0 * u.femtoseconds
        testsystem = testsystems.CustomExternalForcesTestSystem()
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 1)]

    if sysname == "customsplitho":
        timestep = 50.0 * u.femtoseconds
        energy_expressions = ("0.75 * (x^2 + y^2 + z^2)", "0.25 * (x^2 + y^2 + z^2)")
        testsystem = testsystems.CustomPotentialTestSystem(energy_expressions=energy_expressions)
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 2), (1, 1)]

    return system, positions, groups, temperature, timestep, langevin_timestep, testsystem
