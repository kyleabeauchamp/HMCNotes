import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import testsystems

f = lambda x: (x.getPotentialEnergy(), x.getKineticEnergy(), x.getKineticEnergy() + x.getPotentialEnergy())

temperature = 300. * u.kelvin
testsystem = testsystems.LennardJonesFluid(nparticles=2048)
timestep = 0.1 * u.femtoseconds
nsteps = 50

for platform_name in ["CUDA", "OpenCL", "CPU"]:
    platform = mm.Platform.getPlatformByName(platform_name)
    integrators = [mm.VerletIntegrator(timestep), mm.MTSIntegrator(timestep, groups=[(0, 1)])]
    for integrator in integrators:
        print(platform_name, integrator)

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
