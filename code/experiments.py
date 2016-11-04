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
    ############################################################################
    sysname = "switchedljbox"
    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    ############################################################################
    for timestep in [2.5 * u.femtoseconds, 5.0 * u.femtoseconds]:
        collision_rate = 1.0 / u.picoseconds
        integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    collision_rate = None
    for timestep in []:  # [2.0 * u.femtoseconds, 20.0 * u.femtoseconds]:
        integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
        int_string = lb_loader.format_int_name(prms)
        key = (sysname, int_string)
        experiments[key] = integrator

    # hyperopt optimal parameters obtained ~21573 ns / day
    collision_rate = None
    steps_per_hmc = 46
    timestep = 31.928 * u.femtoseconds
    integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    # Using hyperopt optimal parameters from GHMC above, but with manually tuned 2 extra chances
    collision_rate = None
    steps_per_hmc = 46
    timestep = 31.928 * u.femtoseconds
    extra_chances = 2

    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator


    collision_rate = None
    for timestep in []:  # [2.0 * u.femtoseconds]:
        integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
        itype = type(integrator).__name__
        prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
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




    xcghmc_parms = dict(timestep=0.668 * u.femtoseconds, steps_per_hmc=10, extra_chances=1, collision_rate=None)
    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator


    # hyperopt determined optimal settings obtain ~113 effective ns / day
    xcghmc_parms = dict(timestep=1.1868 * u.femtoseconds, steps_per_hmc=23, collision_rate=None, groups=((0, 1), (1, 4)))
    integrator = hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    # Obtained by taking hyperopt optimal GHMCRespa parameters and adding 2 extra chances
    xcghmc_parms = dict(timestep=1.1868 * u.femtoseconds, steps_per_hmc=23, collision_rate=None, extra_chances=2, groups=((0, 1), (1, 4)))
    integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    # hyperopt determined optimal settings obtain ~79.8 effective ns/day
    xcghmc_parms = dict(timestep=0.6791 * u.femtoseconds, steps_per_hmc=20, collision_rate=None)
    integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    xcghmc_parms = dict(timestep=0.6791 * u.femtoseconds, steps_per_hmc=20, collision_rate=None)
    xcghmc_parms.update(dict())
    integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=timestep / u.femtoseconds, collision=lb_loader.fixunits(collision_rate))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator

    ############################################################################
    sysname = "switchedaccuratebigflexiblewater"
    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    ############################################################################

    experiments = OrderedDict()

    # hyperopt determined optimal settings obtain ~113 effective ns / day
    xcghmc_parms = dict(timestep=0.256927 * u.femtoseconds, steps_per_hmc=24, collision_rate=None, groups=((0, 4), (1, 1)))
    integrator = hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, **xcghmc_parms)
    itype = type(integrator).__name__
    prms = dict(sysname=sysname, itype=itype, timestep=integrator.timestep / u.femtoseconds, collision=lb_loader.fixunits(None))
    int_string = lb_loader.format_int_name(prms)
    key = (sysname, int_string)
    experiments[key] = integrator


    return experiments
