from openmmtools import hmc_integrators, testsystems
import pymbar
import pandas as pd
import time
from simtk.openmm import app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u


format_name = lambda prms: "{sysname}_{itype}_{timestep}_{collision}.".format(**prms)


def fixunits(collision_rate):
    if collision_rate is None:
        return -1
    else:
        return collision_rate * u.picoseconds

def remove_cmm(system):
    for k, force in enumerate(system.getForces()):
        ftype = type(force).__name__
        if ftype in ["CMMotionRemover"]:
            system.removeForce(k)
            print("Removed force number %d, %s" % (k, ftype))
            break


def equilibrate(testsystem, temperature, timestep, steps=40000, npt=False, minimize=True, steps_per_hmc=25, use_hmc=False):
    system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

    if npt:
        barostat_index = system.addForce(mm.MonteCarloBarostat(1.0 * u.atmospheres, temperature, 1))
        print(system.getDefaultPeriodicBoxVectors())

    if use_hmc:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
    else:
        integrator = mm.LangevinIntegrator(temperature, 2.0 / u.picoseconds, timestep)

    simulation = build(testsystem, integrator, temperature)

    if minimize:
        simulation.minimizeEnergy()

    integrator.step(steps)

    state = simulation.context.getState(getPositions=True, getParameters=True)
    positions = state.getPositions()
    boxes = state.getPeriodicBoxVectors()

    if use_hmc:
        print(integrator.acceptance_rate)

    if npt:
        system.removeForce(barostat_index)

    system.setDefaultPeriodicBoxVectors(*boxes)  # Doesn't hurt to reset boxes
    print(system.getDefaultPeriodicBoxVectors())

    testsystem.positions = positions
    return positions, boxes


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


def converge(simulation, csv_filename, Neff_cutoff=1E4, sleep_time=0.5 * u.minutes):
    integrator = simulation.integrator
    itype = type(integrator).__name__
    while True:
        simulation.runForClockTime(sleep_time)
        data = pd.read_csv(csv_filename)

        energies = data['Potential Energy (kJ/mole)'].values
        g = pymbar.timeseries.statisticalInefficiency(energies)
        Neff = len(energies) / g

        mu = energies.mean()
        sigma = energies.std()
        stderr = sigma * Neff ** -0.5

        if "HMC" in itype:
            other_str = "arate=%.3f" % integrator.acceptance_rate
        else:
            other_str = ""

        print("t0=%f, energy = %.4f + %.3f, N=%d, g=%.4f, Neff=%.4f, stderr=%f elapsed=%f Neff/s=%f other=%s" % (t0, mu, sigma, len(energies), g, Neff, stderr, elapsed, Neff / elapsed, other_str))

        if Neff > Neff_cutoff:
            return data, g, Neff, mu, sigma, stderr

def build(testsystem, integrator, temperature, precision="mixed"):
    system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': precision}
        simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)
        #context = mm.Context(system, integrator, platform, properties)
    except Exception as e:
        print(e)
        print("Warning: no CUDA platform found, not specifying precision.")
        #context = mm.Context(system, integrator)
        simulation = app.Simulation(topology, system, integrator)

    context = simulation.context
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    return simulation

def load_lj(cutoff=None, dispersion_correction=False, switch_width=None, shift=False, charge=None, ewaldErrorTolerance=None):
    reduced_density = 0.90
    testsystem = testsystems.LennardJonesFluid(nparticles=2048, reduced_density=reduced_density,
    dispersion_correction=dispersion_correction, cutoff=cutoff, switch_width=switch_width, shift=shift, lattice=True, charge=charge, ewaldErrorTolerance=ewaldErrorTolerance)

    system, positions = testsystem.system, testsystem.positions

    timestep = 2 * u.femtoseconds
    langevin_timestep = 0.5 * u.femtoseconds
    xcghmc_timestep = 28.0 * u.femtoseconds

    return testsystem, system, positions, timestep, langevin_timestep

def load(sysname):
    cutoff = 0.9 * u.nanometers
    temperature = 300. * u.kelvin
    langevin_timestep = 0.5 * u.femtoseconds
    timestep = 2 * u.femtoseconds
    equil_steps = 40000
    groups = [(0, 1)]
    steps_per_hmc = 25

    if sysname == "chargedljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(charge=0.15*u.elementary_charge)

    if sysname == "chargedswitchedljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(charge=0.15*u.elementary_charge, switch_width=0.34*u.nanometers)

    if sysname == "chargedlongljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=1.333*u.nanometers, charge=0.15*u.elementary_charge)

    if sysname == "chargedswitchedlongljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=1.333*u.nanometers, charge=0.15*u.elementary_charge, switch_width=0.34*u.nanometers)

    if sysname == "chargedswitchedaccuratelongljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=1.333*u.nanometers, charge=0.15*u.elementary_charge, switch_width=0.34*u.nanometers, ewaldErrorTolerance=5E-5)

    if sysname == "chargedswitchedaccurateljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(charge=0.15*u.elementary_charge, switch_width=0.34*u.nanometers, ewaldErrorTolerance=5E-5)

    if sysname == "ljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj()

    if sysname == "longljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=1.333*u.nanometers)

    if sysname == "shortljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=0.90*u.nanometers)

    if sysname == "shiftedljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(shift=True)

    if sysname == "switchedljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(switch_width=0.34*u.nanometers)

    if sysname == "switchedshortljbox":
        testsystem, system, positions, timestep, langevin_timestep = load_lj(cutoff=0.90*u.nanometers, switch_width=0.34*u.nanometers)

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

    if sysname == "shortwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=0.9*u.nanometers, switch_width=None)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "shortswitchedwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=0.9*u.nanometers, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "water":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=None)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "switchedwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "switchedaccuratewater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms, ewaldErrorTolerance=5E-5)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "switchedaccuratenptwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms, ewaldErrorTolerance=5E-5)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        system.addForce(mm.MonteCarloBarostat(1.0 * u.atmospheres, temperature, 1))

    if sysname == "longswitchedwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.5*u.nanometers, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "rfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=None)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "switchedrfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "longswitchedrfwater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.5*u.nanometers, nonbondedMethod=app.CutoffPeriodic, switch_width=3.0*u.angstroms)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions

    if sysname == "density":
        system, positions = load_lb(hydrogenMass=3.0 * u.amu)
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
        groups = [(0, 4), (1, 1)]
        timestep = 2.0 * u.femtoseconds

    if sysname == "dhfr":
        testsystem = testsystems.DHFRExplicit(nonbondedCutoff=1.1*u.nanometers, nonbondedMethod=app.PME, switch_width=2.0*u.angstroms, ewaldErrorTolerance=5E-5)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=2, others=0)
        groups = [(0, 4), (1, 2), (2, 1)]
        timestep = 0.75 * u.femtoseconds
        equil_steps = 10000

    if sysname == "ho":
        K = 90.0 * u.kilocalories_per_mole / u.angstroms**2
        mass = 39.948 * u.amu
        timestep = np.sqrt(mass / K) * 0.4
        testsystem = testsystems.HarmonicOscillatorArray()
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 1)]

    if sysname == "customho":
        timestep = 1000.0 * u.femtoseconds
        testsystem = testsystems.CustomExternalForcesTestSystem()
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 1)]

    if sysname == "customsplitho":
        timestep = 1000.0 * u.femtoseconds
        energy_expressions = ("0.75 * (x^2 + y^2 + z^2)", "0.25 * (x^2 + y^2 + z^2)")
        testsystem = testsystems.CustomExternalForcesTestSystem(energy_expressions=energy_expressions)
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 2), (1, 1)]

    if sysname == "alanine":
        testsystem = testsystems.AlanineDipeptideImplicit()
        system, positions = testsystem.system, testsystem.positions
        groups = [(0, 2), (1, 1)]
        timestep = 2.0 * u.femtoseconds
        hmc_integrators.guess_force_groups(system, nonbonded=1, others=0)
        #remove_cmm(system)  # Unrolled shouldn't need this
        equil_steps = 10000

    if sysname == "alanineexplicit":
        testsystem = testsystems.AlanineDipeptideExplicit(cutoff=1.1*u.nanometers, switch_width=2*u.angstrom, ewaldErrorTolerance=5E-5)
        system, positions = testsystem.system, testsystem.positions
        #groups = [(0, 2), (1, 1), (2, 1)]
        #groups = [(0, 2), (1, 1)]
        timestep = 1.0 * u.femtoseconds
        #hmc_integrators.guess_force_groups(system, nonbonded=1, others=0, fft=2)
        #hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=1)
        groups = [(0, 1), (1, 2)]
        hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)

        #remove_cmm(system)  # Unrolled doesn't need this
        equil_steps = 4000


    # guess force groups

    if "ljbox" in sysname:
        timestep = 25 * u.femtoseconds
        temperature = 25. * u.kelvin
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1)
        groups = [(0, 2), (1, 1)]
        equil_steps = 10000
        steps_per_hmc = 15


    elif "water" in sysname:
        timestep = 4.0 * u.femtoseconds
        groups = [(0, 1), (1, 2)]
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=0)
        equil_steps = 10000



    return system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc
