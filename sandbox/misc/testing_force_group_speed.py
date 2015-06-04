import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

precision = "mixed"

testsystem = testsystems.DHFRExplicit(nonbondedCutoff=1.1*u.nanometers, nonbondedMethod=app.PME, switch_width=2.0*u.angstroms, ewaldErrorTolerance=5E-5)

system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': precision}
simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)

simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)

hmc_integrators.guess_force_groups(system, nonbonded=1, others=0, fft=2)
del simulation, integrator
timestep = 2.0 * u.femtoseconds
#integrator = mm.LangevinIntegrator(temperature, 2.0 / u.picoseconds, timestep)
#integrator = mm.VerletIntegrator(timestep)
total_steps = 3000
extra_chances = 3
steps_per_hmc = 100
steps = total_steps

steps = total_steps / steps_per_hmc
#integrator = hmc_integrators.GHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep)
integrator = hmc_integrators.XCGHMCIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances)
#integrator = integrators.VelocityVerletIntegrator(2.0 * u.femtoseconds)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision)

integrator.reset_time()
import time
t0 = time.time()
integrator.step(steps)
dt = time.time() - t0
ns_per_day = (timestep / u.nanoseconds) * total_steps / dt * 60 * 60 * 24
dt, ns_per_day
integrator.ns_per_day
