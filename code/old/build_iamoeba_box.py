from sys import stdout
import simtk.openmm as mm
import simtk.openmm.app as app
from simtk import unit as u

pdb = app.PDBFile("/home/kyleb/src/openmm/openmm/wrappers/python/simtk/openmm/app/data/tip3p.pdb")
forcefield = app.ForceField('iamoeba.xml')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*u.nanometers)
system.addForce(mm.MonteCarloBarostat(1*u.atmospheres, 300*u.kelvin, 25))

integrator = mm.LangevinIntegrator(300 * u.kelvin, 1.0 / u.picoseconds, 0.5*u.femtoseconds)

platform = mm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300*u.kelvin)
print('Equilibrating...')
simulation.step(100)

simulation.reporters.append(app.DCDReporter('./iamoeba/trajectory.dcd', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=1000, separator='\t'))

simulation.step(100000)

print('Running Production...')
