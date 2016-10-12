import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

f = lambda x: (x.getPotentialEnergy(), x.getKineticEnergy(), x.getKineticEnergy() + x.getPotentialEnergy())


platform_name = "OpenCL"

temperature = 300. * u.kelvin
testsystem = testsystems.LennardJonesFluid(nparticles=2048)
timestep = 0.01 * u.femtoseconds
nsteps = 10
steps_per_hmc = 1
groups = [(0, 1)]
collision_rate = None

integrators = [
hmc_integrators.GHMCRESPAIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, groups=groups, collision_rate=collision_rate),
hmc_integrators.GHMCIntegrator(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate),
]

platform = mm.Platform.getPlatformByName(platform_name)

for integrator in integrators:
    print(integrator)

    simulation = app.Simulation(testsystem.topology, testsystem.system, integrator, platform=platform)
    simulation.context.setPositions(testsystem.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    integrator.step(nsteps)  # Do a single step first to apply constraints before we run tests.

    state1a = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True, getVelocities=True)
    old = f(state1a)
    print("PE, KE, TOTAL")
    print(old)
    integrator.step(nsteps)
    state1b = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True, getVelocities=True)
    new = f(state1b)
    print(new)
    print("delta = %s" % (new[-1] - old[-1]))
    print(integrator.acceptance_rate)
