import sklearn.grid_search
import scipy.stats.distributions
import lb_loader
import simtk.openmm.app as app
import numpy as np
import pandas as pd
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
pd.set_option('display.width', 1000)

n_steps = 1500
temperature = 300. * u.kelvin

hydrogenMass = 3.0 * u.amu

testsystem = testsystems.DHFRExplicit(hydrogenMass=hydrogenMass, nonbondedCutoff=1.0 * u.nanometers)
system, positions = testsystem.system, testsystem.positions

hmc_integrators.guess_force_groups(system, nonbonded=1, fft=2, others=0)

integrator = mm.LangevinIntegrator(temperature, 1.0 / u.picoseconds, 0.25 * u.femtoseconds)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

%time integrator.step(5000)
positions = context.getState(getPositions=True).getPositions()
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy() + state.getKineticEnergy()
energy, state.getPotentialEnergy(), state.getKineticEnergy()

collision_rate = 10000.0 / u.picoseconds


possible_respa_schemes = [
dict(nonbonded=1, fft=1, others=1),
dict(nonbonded=1, fft=1, others=2),
dict(nonbonded=1, fft=1, others=3),
dict(nonbonded=1, fft=1, others=4),
dict(nonbonded=1, fft=1, others=5),
dict(nonbonded=2, fft=1, others=2),
dict(nonbonded=3, fft=1, others=3),
dict(nonbonded=2, fft=1, others=4),
]

def set_respa(system, scheme):
    """INPLACE on system!"""
    if len(scheme.values()) == 1:
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=0, others=0)
        groups = [(0, 1)]
    if len(scheme.values()) == 2:
        hmc_integrators.guess_force_groups(system, nonbonded=0, fft=1, others=0)
        groups = [(0, scheme["others"]), (1, scheme["nonbonded"])]
    if len(scheme.values()) == 3:
        hmc_integrators.guess_force_groups(system, nonbonded=1, fft=2, others=0)
        groups = [(0, scheme["others"]), (1, scheme["nonbonded"]), (2, scheme["fft"])]
    return groups

unigen = lambda min, max: scipy.stats.uniform(loc=min, scale=(max - min))
#min_timestep = 1.0
#max_timestep = 3.0
min_timestep = 2.0
max_timestep = 2.75

possible_respa_schemes = [
dict(nonbonded=1, fft=1, others=3),
]

min_steps_per_hmc = 15
max_steps_per_hmc = 25


params_grid = {
"steps_per_hmc" : scipy.stats.distributions.randint(low=min_steps_per_hmc, high=max_steps_per_hmc),
"timestep" : unigen(min_timestep, max_timestep),
"respa": possible_respa_schemes,
}



def hmc_inner(system, positions, params):
    groups = set_respa(system, params["respa"])
    integrator = hmc_integrators.GHMCRESPA(temperature, params["steps_per_hmc"], params["timestep"] * u.femtoseconds, collision_rate, groups)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(n_steps)
    d = integrator.summary
    d.update(params)
    print(pd.Series(d))
    print("**************************** %f" % integrator.effective_ns_per_day)
    return d

data = []
n_iter = 10
params_list = list(sklearn.grid_search.ParameterSampler(params_grid, n_iter=n_iter))
for params in params_list:
    data.append(hmc_inner(system, positions, params))

X = pd.DataFrame(data)
X.iloc[:, 3:]

X[X.respa == {u'fft': 1, u'nonbonded': 1, u'others': 2}].iloc[:, 3:]
X[X.respa == {u'fft': 1, u'nonbonded': 1, u'others': 3}].iloc[:, 3:]




k_max = 10
#timestep = 2.33 * u.femtoseconds
timestep = 2.5 * u.femtoseconds
steps_per_hmc = 23
groups = set_respa(system, dict(nonbonded=1, fft=1, others=3))

integrator = hmc_integrators.XHMCRESPAIntegrator(temperature, steps_per_hmc, timestep, collision_rate, k_max, groups)
context = mm.Context(system, integrator)
context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

integrator.step(2500)
data = integrator.vstep(25)
integrator.effective_timestep, integrator.effective_ns_per_day
