import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

platform_name = "CUDA"
precision = "mixed"

sysname = "switchedljbox"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

int_args = dict(
temperature = temperature,
timestep = 1.0,
steps_per_hmc = 1,
collision_rate = 1.0 / u.picoseconds,
extra_chances = 0,
group0_iterations = 2,
groups=(0,1,1),
)

collision_rate = int_args["collision_rate"]

groups = int_args["groups"]
hmc_integrators.guess_force_groups(system, others=groups[0], nonbonded=groups[1], fft=groups[2])

steps = 1000

integrator = lb_loader.kw_to_int(**int_args)
#integrator = integrators.HMCIntegrator(temperature=temperature, nsteps=steps_per_hmc, timestep=timestep)
#integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)

#temperature = int_args["temperature"]
#steps_per_hmc = int_args["steps_per_hmc"]
#timestep = int_args["timestep"]
groups = [(0, 1)]
extra_chances = 1
#collision_rate = None
integrator = hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
#integrator = hmc_integrators.GHMCRESPAIntegrator(temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups)
#integrator = hmc_integrators.XCGHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, groups=groups, collision_rate=collision_rate)
#integrator = hmc_integrators.XCGHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, extra_chances=extra_chances, collision_rate=collision_rate)

simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name, state=state)

integrator.step(steps)

print(integrator)
print(integrator.effective_timestep, integrator.effective_ns_per_day)
print(integrator.acceptance_rate)
