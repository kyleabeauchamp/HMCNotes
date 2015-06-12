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

simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)

%time simulation.runForClockTime(0.1 * u.minutes)
