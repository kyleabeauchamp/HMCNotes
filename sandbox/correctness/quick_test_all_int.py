import lb_loader
import pandas as pd
import simtk.openmm.app as app
import numpy as np
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import integrators, testsystems
pd.set_option('display.width', 1000)

groups = [(0, 1)]
sysname = "ljbox"

system, positions, groups, temperature, timestep = lb_loader.load(sysname)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep / 4.)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(1000)
positions = context.getState(getPositions=True).getPositions()

timestep = 1.0 * u.femtoseconds
extra_chances = 1


integrator = integrators.HMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep

integrator = integrators.GHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep

integrator = integrators.XCGHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=extra_chances, take_debug_steps=False)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep

integrator = integrators.XCGHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, extra_chances=extra_chances, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep

integrator = integrators.XCHMCIntegrator(temperature, steps_per_hmc=10, timestep=timestep, extra_chances=extra_chances, take_debug_steps=False)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep

integrator = integrators.XCHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, extra_chances=extra_chances, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep


integrator = integrators.HMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep


integrator = integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=10, timestep=timestep, collision_rate=1.0 / u.picoseconds, groups=groups)
context = lb_loader.build(system, integrator, positions, temperature)
integrator.step(40)
output = integrator.vstep(10)
integrator.effective_timestep
