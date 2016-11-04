import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators
from sys import stdout

platform_name = "CUDA"
platform = mm.Platform.getPlatformByName(platform_name)
properties = {'CudaPrecision': "mixed"}
precision = "mixed"

#sysname = "switchedaccurateflexiblewater"
sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

n_steps = 20000
temperature = 300. * u.kelvin
timestep = 1.75 * u.femtoseconds




for integrator in [integrators.PositionVerletIntegrator(timestep), integrators.VelocityVerletIntegrator(timestep)]:
    intname = integrator.__class__.__name__
    print("*" * 80)
    print(sysname, intname)
    csv_filename = "./verlet/%s_%s_%f.csv" % (sysname, intname, timestep / u.femtoseconds)
    simulation = app.Simulation(testsystem.topology, testsystem.system, integrator, platform=platform, platformProperties=properties)
    simulation.context.setPositions(testsystem.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(1000)
    #simulation.reporters.append(app.StateDataReporter(csv_filename, 5, step=True, time=True, potentialEnergy=True, kineticEnergy=False, totalEnergy=True, elapsedTime=True))
    import time
    t0 = time.time()
    simulation.step(n_steps)
    dt = time.time() - t0
    print(dt)
    del simulation, integrator
