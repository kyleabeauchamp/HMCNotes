import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

n_steps = 400
platform_name = "CUDA"
precision = "mixed"

sysname = "amoeba"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

#timestep = 0.6791 * u.femtoseconds
#steps_per_hmc = 20

timestep = 0.5 * u.femtoseconds
steps_per_hmc = 31
#extra_chances = 5
collision_rate = None
groups = ((0, 1), (1, 2))

steps = 20

integrator = hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)
simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)

integrator.step(steps)

#print(integrator.all_probs)
print(integrator.effective_timestep, integrator.effective_ns_per_day)
print(integrator.acceptance_rate)
