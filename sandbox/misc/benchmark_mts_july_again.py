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
timestep = 1.0 * u.femtoseconds
steps = 5000

#hmc_integrators.guess_force_groups(system, nonbonded=0, others=1, fft=0)
#groups = [(0, 1), (1, 4)]

hmc_integrators.guess_force_groups(system, nonbonded=0, others=0, fft=0)
groups = [(0, 1)]


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


print groups
print mm.version.full_version, dt, ns_per_day, outer_per_sec, inner_per_sec



"""
[(0, 1)]
6.3.0.dev-9b345f2 12.7207889557 33.9601577783 393.057381694 393.057381694

[(0, 1), (1, 4)]
6.3.0.dev-9b345f2 14.9431400299 28.9095865484 334.601696162 1338.406784


[(0, 1)]
6.3.0.dev-637de3d 13.2098071575 32.7029755127 378.506661027 378.506661027

[(0, 1), (1, 4)]
6.3.0.dev-637de3d 15.7206330299 27.4798094439 318.053350045 1272.21340018


"""
