import lb_loader
import simtk.openmm.app as app
import simtk.openmm as mm
from simtk import unit as u
import sys
from experiments import enumerate_experiments
import pickle

def run():


    experiments = enumerate_experiments()
    for namestr, integrator in experiments.items():
        integrator_filename = "./data/systems/%s_system.xml" % namestr
        pickle.dump(open(integrator_filename, 'w'), integrator)

if __name__ == "__main__":
    run()
