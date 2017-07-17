#!/usr/bin/env python
import lb_loader
import simtk.openmm as mm
import pickle
import fire


def run(sysname, system_filename, state_filename):

    system, positions, groups, temperature, timestep, langevin_timestep, testsystem, equil_steps, steps_per_hmc = lb_loader.load(sysname)
    positions, boxes, state = lb_loader.equilibrate(testsystem, temperature, langevin_timestep, steps=equil_steps, minimize=True)

    pickle.dump(testsystem, open(system_filename, 'wb'))
    serialized = mm.XmlSerializer.serialize(state)
    lb_loader.write_file(state_filename, serialized)


if __name__ == "__main__":
    fire.Fire(run)
