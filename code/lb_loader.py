from openmmtools import hmc_integrators, testsystems
import pymbar
import pandas as pd
from simtk.openmm import app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u

PLATFORM = "CUDA"
PRECISION = "mixed"

format_int_name = lambda prms: "{itype}_{timestep}_{collision}".format(**prms)
format_name = lambda prms: "{sysname}/{itype}_{timestep}_{collision}.".format(**prms)

num_to_groups = lambda group0_iterations, n_groups: ([(0, group0_iterations)] + [(i, 1) for i in range(1, n_groups)])

def kw_to_int(**kwargs):
    steps_per_hmc = kwargs["steps_per_hmc"]
    timestep = kwargs["timestep"]
    group0_iterations = kwargs["group0_iterations"]
    temperature = kwargs["temperature"]
    print("*" * 80)
    print("steps=%d, timestep=%f, extra_chances=%d, group0_iterations=%d" % (steps_per_hmc, timestep, 0, group0_iterations))
    timestep = timestep * u.femtoseconds
    steps_per_hmc = int(steps_per_hmc)
    group0_iterations = int(group0_iterations)
    n_groups = max(kwargs["groups"]) + 1  # 1 group means group number = 0, assumes zero index and ordering
    groups = num_to_groups(group0_iterations, n_groups)
    print(n_groups)
    print(groups)

    if kwargs["group0_iterations"] == 0 and kwargs["extra_chances"] == 0:
        integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
    elif kwargs["group0_iterations"] > 0 and kwargs["extra_chances"] > 0 :
        integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)
    elif kwargs["group0_iterations"] > 0 and kwargs["extra_chances"] == 0 :
        integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)
    elif kwargs["group0_iterations"] == 0 and kwargs["extra_chances"] > 0 :
        integrator = hmc_integrators.XCGHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)

    return integrator

def write_file(filename, contents):
    with open(filename, 'w') as outfile:
        outfile.write(contents)


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


def equilibrate(testsystem, temperature, timestep, steps=40000, npt=False, minimize=True, steps_per_hmc=25, use_hmc=False, platform_name=PLATFORM, precision=PRECISION):
    system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

    if npt:
        barostat_index = system.addForce(mm.MonteCarloBarostat(1.0 * u.atmospheres, temperature, 1))
        print(system.getDefaultPeriodicBoxVectors())

    if use_hmc:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
    else:
        integrator = mm.LangevinIntegrator(temperature, 2.0 / u.picoseconds, timestep)

    simulation = build(testsystem, integrator, temperature, platform_name=platform_name, precision=precision)

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

    testsystem.positions = positions
    return positions, boxes, state


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


def summarize(filename):
    statedata = pd.read_csv(filename)
    energies = statedata["Potential Energy (kJ/mole)"].values
    if "Kinetic Energy (kJ/mole)" in statedata.columns:
        holding = pd.read_csv(filename)["Kinetic Energy (kJ/mole)"].values
        mu = holding.dot(energies) / holding.sum()
        mu = energies.mean()
    else:
        mu = energies.mean()
    g = pymbar.timeseries.statisticalInefficiency(energies)
    Neff = max(1, len(energies) / g)
    sigma = energies.std()
    stderr = sigma * Neff ** -0.5
    walltime = statedata["Elapsed Time (s)"].max()
    neffs = Neff / walltime

    return statedata, pd.Series(dict(filename=filename, g=g, Neff=Neff, mu=mu, sigma=sigma, stderr=stderr, walltime=walltime, Neff_per_sec=neffs))


def converge(simulation, csv_filename, Neff_cutoff=1E4, sleep_time=0.5 * u.minutes):
    while True:
        simulation.runForClockTime(sleep_time)
        statedata, results = summarize(csv_filename)
        print(results.to_frame().T)
        if results.Neff > Neff_cutoff:
            return statedata, results


def build(testsystem, integrator, temperature, precision="mixed", state=None, platform_name="CUDA"):
    system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            properties = {'CudaPrecision': precision, "DeterministicForces": "1"}
            simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)
        if platform_name == "OpenCL":
            properties = {'Precision': precision}
            simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)
        elif platform_name == "CPU":
            simulation = app.Simulation(topology, system, integrator, platform=platform)
    except Exception as e:
        print(e)
        print("Warning: no CUDA platform found, not specifying precision.")
        simulation = app.Simulation(topology, system, integrator)

    context = simulation.context
    context.setPositions(positions)

    if state is not None:
        context.setState(state)

    context.setVelocitiesToTemperature(temperature)

    return simulation

def load_lj(cutoff=None, dispersion_correction=False, switch_width=None, shift=False, charge=None, ewaldErrorTolerance=None):
    reduced_density = 0.90
    testsystem = testsystems.LennardJonesFluid(nparticles=2048, reduced_density=reduced_density,
    dispersion_correction=dispersion_correction, cutoff=cutoff, switch_width=switch_width, shift=shift, lattice=True, charge=charge, ewaldErrorTolerance=ewaldErrorTolerance)

    system, positions = testsystem.system, testsystem.positions

    timestep = 2 * u.femtoseconds
    langevin_timestep = 0.5 * u.femtoseconds
    # optimal xcghmc parms determined via hyperopt
    xcghmc_parms = dict(timestep=32.235339 * u.femtoseconds, steps_per_hmc=16, extra_chances=1, collision_rate=None)

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
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)  # We may want to try reduce

    if sysname == "switchedaccurateflexiblewater":
        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms, ewaldErrorTolerance=5E-5, constrained=False)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        # Using these groups for hyperopt-optimal RESPA integrators
        #hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
        hmc_integrators.guess_force_groups(system, nonbonded=1, others=1, fft=0)
        equil_steps = 100000
        langevin_timestep = 0.25 * u.femtoseconds

    if sysname == "switchedaccuratebigflexiblewater":
        testsystem = testsystems.WaterBox(box_edge=10.0 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms, ewaldErrorTolerance=5E-5, constrained=False)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        # Using these groups for hyperopt-optimal RESPA integrators
        hmc_integrators.guess_force_groups(system, nonbonded=1, others=0, fft=1)
        equil_steps = 100000
        langevin_timestep = 0.1 * u.femtoseconds


    if sysname == "switchedaccuratestiffflexiblewater":

        testsystem = testsystems.WaterBox(box_edge=3.18 * u.nanometers, cutoff=1.1*u.nanometers, switch_width=3.0*u.angstroms, ewaldErrorTolerance=5E-5, constrained=False)  # Around 1060 molecules of water
        system, positions = testsystem.system, testsystem.positions
        # Using these groups for hyperopt-optimal RESPA integrators
        #hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
        hmc_integrators.guess_force_groups(system, nonbonded=1, others=1, fft=0)
        equil_steps = 100000
        langevin_timestep = 0.25 * u.femtoseconds

        FACTOR = 10.0
        # Make bonded terms stiffer to highlight RESPA advantage
        for force in system.getForces():
            if type(force) == mm.HarmonicBondForce:
                for i in range(force.getNumBonds()):
                    (a0, a1, length, strength) = force.getBondParameters(i)
                    force.setBondParameters(i, a0, a1, length, strength * FACTOR)

            elif type(force) == mm.HarmonicAngleForce:
                for i in range(force.getNumAngles()):
                    (a0, a1, a2, length, strength) = force.getAngleParameters(i)
                    force.setAngleParameters(i, a0, a1, a2, length, strength * FACTOR)




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


    if sysname == "src":
        testsystem = testsystems.SrcExplicit(nonbondedCutoff=1.1*u.nanometers, nonbondedMethod=app.PME, switch_width=2.0*u.angstroms, ewaldErrorTolerance=5E-5)
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=1, others=0)
        groups = [(0, 2), (1, 1)]
        timestep = 0.25 * u.femtoseconds
        equil_steps = 1000

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

    if sysname == "amoeba":
        testsystem = testsystems.AMOEBAIonBox()
        system, positions = testsystem.system, testsystem.positions
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)


    return system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc
