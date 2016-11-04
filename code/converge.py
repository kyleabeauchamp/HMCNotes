#!/usr/bin/env python
import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
from openmmtools import hmc_integrators, testsystems
from experiments import enumerate_experiments
import pickle
import sys

def run():
    system_filename, state_filename, integrator_filename, sysname, Neff_cutoff, csv_filename, dcd_filename = sys.argv[1:]

    Neff_cutoff = float(Neff_cutoff)

    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)

    state = mm.XmlSerializer.deserialize(open(state_filename).read())
    testsystem = pickle.load(open(system_filename, 'rb'))
    integrator = pickle.load(open(integrator_filename, 'rb'))

    itype = type(integrator).__name__
    print(itype)

    simulation = lb_loader.build(testsystem, integrator, temperature, state=state)
    simulation.runForClockTime(1.0 * u.minutes)

    output_frequency = 100 if "Langevin" in itype else 1
    kineticEnergy = True if "MJHMC" in itype else False
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, time=True, potentialEnergy=True, kineticEnergy=kineticEnergy, temperature=True, density=True, elapsedTime=True))
    simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
    lb_loader.converge(simulation, csv_filename, Neff_cutoff)

if __name__ == "__main__":

    run()
