import experiments

experiments = experiments.enumerate_experiments()
DCD_FILENAMES = ["data/{sysname}/{intname}.dcd".format(sysname=sysname, intname=intname) for (sysname, intname) in experiments.keys()]

N_EFF = 100000

rule all:
    input:
        dcd=DCD_FILENAMES,

rule serialize_integrators:
    output:
        integrator="data/integrators/{sysname}/{integrator}.pkl",
    shell:
        "serialize_integrators.py {wildcards.sysname} {wildcards.integrator} {output.integrator}"

rule serialize_systems:
    output:
        system="data/systems/{sysname}_state.xml",
        state="data/systems/{sysname}_system.pkl",
    shell:
        "serialize_system.py {wildcards.sysname} {output.system} {output.state};"

rule run_production:
    input:
        system="data/systems/{sysname}_state.xml",
        state="data/systems/{sysname}_system.pkl",
        integrator="data/integrators/{sysname}/{integrator}.pkl",
    output:
        dcd="data/{sysname}/{integrator}.dcd",
        csv="data/{sysname}/{integrator}.csv",
    shell:
        "converge.py {input.system} {input.state} {input.integrator} {wildcards.sysname} {N_EFF} {output.csv} {output.dcd}"
