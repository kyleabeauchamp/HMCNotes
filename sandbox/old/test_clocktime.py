import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import testsystems

testsystem = testsystems.DHFRExplicit()

system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': "mixed"}

temperature = 1 * u.kelvin
timestep = 0.5 * u.femtoseconds

integrator = mm.VerletIntegrator(timestep)

simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)
csv_filename = "./test.csv"
output_frequency = 1

simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)
integrator.step(1)  # Pre-build the customintegrator if applicable
simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, temperature=True, density=True, elapsedTime=True))

simulation.runForClockTime(1 * u.seconds)
