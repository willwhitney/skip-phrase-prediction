import os
import sys

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv
gpu     = '--gpu' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


networks_prefix = "networks"

base_networks = {
    }


free_gpus = [0,2,3,4,5,6,7]

jobs = []

learning_rate_options = [3e-3, 1e-3, 1e-4]
spacing_options = [
        (10,1,10),
        # (10,5,10),
        (10,10,10),
    ]

for learning_rate in learning_rate_options:
    for spacing in spacing_options:
        job = {
                "n_context": spacing[0],
                "n_skip": spacing[1],
                "n_predict": spacing[2],
                "learning_rate": learning_rate,
            }
        jobs.append(job)

if gpu:
    if len(jobs) > len(free_gpus):
        raise Exception("More GPU jobs specified than GPUs.")
    for i, job in enumerate(jobs):
        job['gpuid'] = free_gpus[i]

if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "atari"
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " --" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " --" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " --" + flag + " " + networks_prefix + "/" + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " --" + flag + " " + str(job[flag])
    flagstring = flagstring + " --name " + jobname

    jobcommand = "th main.lua" + flagstring

    print(jobcommand)
    if local
        if not dry_run:
            if detach:
                os.system(jobcommand + ' 2> logs/' + jobname + '.err 1> logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write(jobcommand)

        if not dry_run:
            if 'gpu' in job and job['gpu']:
                os.system("sbatch -N 1 -c 2 --gres=gpu:1 -p gpu --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
            else:
                os.system("sbatch -N 1 -c 2 --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
