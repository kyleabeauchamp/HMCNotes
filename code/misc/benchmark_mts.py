import time
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems, integrators

#testsystem = testsystems.DHFRExplicit()
testsystem = testsystems.DHFRExplicit(rigid_water=False, constraints=None)

system, topology, positions = testsystem.system, testsystem.topology, testsystem.positions

platform = mm.Platform.getPlatformByName('CUDA')
#properties = {}
properties = {'CudaPrecision': "single"}

temperature = 1 * u.kelvin
timestep = 4.0 * u.femtoseconds
steps = 5000

#hmc_integrators.guess_force_groups(system, nonbonded=1, others=2, fft=0)
#groups = [(0, 1), (1, 2), (2, 4)]  # 59.3 ns / day

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
#groups = [(0, 1), (1, 1)]  # 79.4 ns / day

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
#groups = [(0, 1), (1, 2)]  # 98.5 ns / day

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(0, 1), (1, 2)]  # 94.2

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(0, 1), (1, 4)]  # 78.1 CUDA, 0.317 CPU

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(1, 4)]  # Just update velocities and positions

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(0, 1)]  # 111.6 ns / day CUDA, 1.065 CPU

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
#groups = [(0, 1)]

hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
groups = [(0, 1), (1, 4)]


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
