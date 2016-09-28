import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
from collections import OrderedDict

def get_grid(sysname, temperature, timestep, langevin_timestep, groups, steps_per_hmc=100, extra_chances=5):

    integrators = OrderedDict()

    for timestep in [0.4 * langevin_timestep, 2.0 * langevin_timestep]:
        collision_rate = 1.0 / u.picoseconds
        integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        fmt_string = lb_loader.format_name(prms)
        integrators[fmt_string] = integrator

    collision_rate = None
    for timestep in [timestep, 23.8 * u.femtoseconds]:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        fmt_string = lb_loader.format_name(prms)
        integrators[fmt_string] = integrator

    collision_rate = None
    for timestep in [0.5 * timestep, 2.0 * timestep]:
        integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        fmt_string = lb_loader.format_name(prms)
        integrators[fmt_string] = integrator

    #hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate))
    #hmc_integrators.MJHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, beta_mixing=0.25)

    return integrators

walltime = 1.0 * u.hours
sysname = "switchedljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes = lb_loader.equilibrate(testsystem, temperature, timestep, steps=equil_steps, minimize=True)

for fmt_string, integrator in get_grid(sysname, temperature, timestep, langevin_timestep, groups).items():
    itype = type(integrator).__name__
    print("%s    %s" % (fmt_string, itype))

    csv_filename = "./data/%s.csv" % fmt_string
    pdb_filename = "./data/%s.pdb" % fmt_string
    dcd_filename = "./data/%s.dcd" % fmt_string


    simulation = lb_loader.build(testsystem, integrator, temperature)
    simulation.step(5)

    output_frequency = 100 if "Langevin" in itype else 2
    kineticEnergy = True if "MJHMC" in itype else False
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, kineticEnergy=kineticEnergy, temperature=True, density=True, elapsedTime=True))
    simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
    simulation.runForClockTime(walltime)
