#!/usr/bin/env python
import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
import sys
from experiments import enumerate_experiments
import pickle

def run():
    sysname, intname, integrator_filename = sys.argv[1:]
    experiments = enumerate_experiments()
    integrator = experiments[(sysname, intname)]
    pickle.dump(integrator, open(integrator_filename, 'wb'))

if __name__ == "__main__":
    run()
