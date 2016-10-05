import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
from collections import OrderedDict


def enumerate_experiments():
    experiments = OrderedDict()
    extra_chances = 3
    ############################################################################
    sysname = "switchedljbox"
    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    ############################################################################
    for timestep in [0.5 * u.femtoseconds, 2.0 * u.femtoseconds]:
        collision_rate = 1.0 / u.picoseconds
        integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    collision_rate = None
    for timestep in [2.0 * u.femtoseconds, 20.0 * u.femtoseconds]:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    collision_rate = None
    for timestep in [2.0 * u.femtoseconds]:
        integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator


    xcghmc_parms = dict(timestep=32.2 * u.femtoseconds, steps_per_hmc=16, extra_chances=1, collision_rate=None)
    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    ############################################################################
    sysname = "switchedaccurateflexiblewater"
    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    ############################################################################

    for timestep in [0.10 * u.femtoseconds, 0.15 * u.femtoseconds, 0.5 * u.femtoseconds]:
        collision_rate = 1.0 / u.picoseconds
        integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    collision_rate = None
    for timestep in [0.4 * u.femtoseconds, 0.5 * u.femtoseconds]:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    collision_rate = None
    timestep = 0.5 * u.femtoseconds
    steps_per_hmc=100
    extra_chances=5
    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator


    xcghmc_parms = dict(timestep=0.668 * u.femtoseconds, steps_per_hmc=10, extra_chances=1, collision_rate=None)
    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    return experiments
