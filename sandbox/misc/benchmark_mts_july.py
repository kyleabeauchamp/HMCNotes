import time
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

testsystem = testsystems.DHFRExplicit(rigid_water=False, constraints=None)

system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': "single"}

temperature = 1 * u.kelvin
timestep = 2.0 * u.femtoseconds
steps = 5000

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(0, 1)]
# new (65.32461094856262, 13.226255578932173, 76.54083089659821, 76.54083089659821)
# old (64.60578417778015, 13.373415569455394, 77.39245121212612, 77.39245121212612)

hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
groups = [(0, 1), (1, 4)]
# new (69.01602101325989, 12.518832400291547, 72.44694676094645, 289.7877870437858)
# old (68.56146907806396, 12.601830322746602, 72.92725881219098, 291.70903524876394)


integrator = mm.MTSIntegrator(timestep, groups)
simulation = app.Simulation(topology, system, integrator, platform=platform, platformProperties=properties)

simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)

integrator.step(1)

t0 = time.time()
integrator.step(steps)
dt = time.time() - t0
outer_per_day = steps / dt * 60 * 60 * 24
outer_per_sec = steps / dt
inner_per_sec = outer_per_sec * groups[-1][1]
ns_per_day = (timestep / u.nanoseconds) * outer_per_day


dt, ns_per_day, outer_per_sec, inner_per_sec
