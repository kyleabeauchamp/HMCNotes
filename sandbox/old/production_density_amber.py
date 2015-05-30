from openmmtools import hmc_integrators, testsystems
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit as u

timestep = 1.0 * u.femtoseconds

target_length = 200 * u.nanoseconds
steps_per_hmc = 12

n_steps = int(target_length / timestep)
output_frequency = 20
#barostat_frequency = 4
barostat_frequency = 1

print(n_steps, output_frequency, barostat_frequency)

collision_rate = 1.0 / u.picoseconds
temperature = 300. * u.kelvin
pressure = 1.0 * u.atmospheres

cutoff = 0.95 * u.nanometers

prmtop_filename = "./input/126492-54-4_1000_300.6.prmtop"  # cp ~/src/kyleabeauchamp/LiquidBenchmark/liquid_benchmark_3_14/tleap/126492-54-4_1000_300.6.prmtop  ./
pdb_filename = "./input/126492-54-4_1000_300.6_equil.pdb"  # cp ~/src/kyleabeauchamp/LiquidBenchmark/liquid_benchmark_3_14/equil/126492-54-4_1000_300.6_equil.pdb ./

log_filename = "./production/production_%0.2f.log" % (timestep / u.femtoseconds)


pdb = app.PDBFile(pdb_filename)
prmtop = app.AmberPrmtopFile(prmtop_filename)

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff, constraints=app.HBonds)

#integrator = hmc_integrators.GHMC2(temperature, steps_per_hmc, timestep, collision_rate)
integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, timestep)
#system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

simulation = app.Simulation(prmtop.topology, system, integrator)

simulation.context.setPositions(pdb.positions)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())
simulation.context.setVelocitiesToTemperature(temperature)

simulation.step(100)

simulation.reporters.append(app.StateDataReporter(open(log_filename, 'w'), output_frequency, step=True, time=True, speed=True, density=True, potentialEnergy=True, kineticEnergy=True))
#simulation.step(n_steps)
simulation.step(1000)
