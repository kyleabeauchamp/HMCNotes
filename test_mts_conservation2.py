import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
from sys import stdout

platform_name = "CUDA"
platform = mm.Platform.getPlatformByName(platform_name)
properties = {'CudaPrecision': "mixed"}
precision = "mixed"

sysname = "switchedaccurateflexiblewater"
#sysname = "switchedaccuratewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

n_steps = 100000
temperature = 300. * u.kelvin
timestep = 1.75 * u.femtoseconds

step_groups = [
(0, 0),
(1, 1),
(1, 2),
(1, 3),
(2, 1),
(3, 1),
]

for i, (step0, step1) in enumerate(step_groups):
    groups = ((0, step0), (1, step1))
    if step0 == 0 and step1 == 0:
        integrator = mm.VerletIntegrator(timestep)
    else:
        integrator = mm.MTSIntegrator(timestep, groups)
    print("*" * 80)
    print(i, sysname, step0, step1)
    csv_filename = "./conservation/%s_%f_%d_%d.csv" % (sysname, timestep / u.femtoseconds, step0, step1)
    simulation = app.Simulation(testsystem.topology, testsystem.system, integrator, platform=platform, platformProperties=properties)
    simulation.context.setPositions(testsystem.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(1000)
    simulation.reporters.append(app.StateDataReporter(csv_filename, 5, step=True, time=True, potentialEnergy=True, kineticEnergy=False, totalEnergy=True, elapsedTime=True))
    simulation.step(n_steps)
    del simulation, integrator
