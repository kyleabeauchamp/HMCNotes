import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems

f = lambda x: (x.getPotentialEnergy(), x.getKineticEnergy(), x.getKineticEnergy() + x.getPotentialEnergy())
platform_name = "CUDA"
precision = "mixed"
sysname = "switchedaccurateflexiblewater"

system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True, use_hmc=False, precision=precision, platform_name=platform_name)

n_iter = 1000
n_steps = 25
temperature = 300. * u.kelvin
timestep = 1.0 * u.femtoseconds

x = []

integrators = [mm.VerletIntegrator(timestep),   # 0
               mm.MTSIntegrator(timestep, groups=((0, 1), (1, 1))),  # 1
               mm.MTSIntegrator(timestep, groups=((0, 2), (1, 1))),  # 2
               mm.MTSIntegrator(timestep, groups=((0, 3), (1, 1))),  # 3
               mm.MTSIntegrator(timestep, groups=((0, 4), (1, 1))),  # 4
               mm.MTSIntegrator(timestep, groups=((0, 5), (1, 1))),  # 5
               mm.MTSIntegrator(timestep, groups=((1, 1), (0, 1))),  # 6
               mm.MTSIntegrator(timestep, groups=((1, 3), (0, 1))),  # 7
               mm.MTSIntegrator(timestep, groups=((1, 2), (0, 1))),  # 8
               mm.MTSIntegrator(timestep, groups=((1, 4), (0, 1))),  # 9
               mm.MTSIntegrator(timestep, groups=((1, 5), (0, 1))),  # 10
               ]
for (i, integrator) in enumerate(integrators):
    print("*" * 80)
    simulation = lb_loader.build(testsystem, integrator, temperature, precision=precision, platform_name=platform_name)
    integrator.step(n_steps * 2)  # Pre-equilibrate with chosen integrator
    for k in range(n_iter):
        print(i, k)
        state1a = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True, getVelocities=True)
        old = f(state1a)
        print("PE, KE, TOTAL")
        print(old)
        integrator.step(n_steps)
        state1b = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True, getVelocities=True)
        new = f(state1b)
        print(new)
        delta = (new[-1] - old[-1])
        print("delta = %s" % delta)
        x.append(dict(INT=i, RUN=k, VAL=delta))

x = pd.DataFrame(x)
x["VAL"] = x.VAL.map(lambda x: x / x.unit)
print(x.groupby(["INT"]).VAL.mean())
