#!/usr/bin/env python
import fire
from experiments import enumerate_experiments, EXPERIMENTS
import pickle


def run(sysname, intname, integrator_filename):
    enumerate_experiments()
    integrator = EXPERIMENTS[(sysname, intname)].integrator
    pickle.dump(integrator, open(integrator_filename, 'wb'))


if __name__ == "__main__":
    fire.Fire(run)
