import subprocess
import json

mock_dirs = ["../mock_1/", "../mock_2/", "../mock_3/"]

config = {
    "location": "server",
    "a_stretch": 0.8,
    "nwalk": 15,
    "nstep": 100000,
    "ncores": 16,
    "min_mass": 6.5,
    "max_mass":11.5,
    "Nsamp": 1,
    "max_N": 1000,
    "init_gauss": 1e-3,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
    }

for idx, dir in enumerate(mock_dirs, start=0):

    config["savedir"] = dir
    config["start_theta"] = [10.5, 2.0, 0, 0, 0.2, 0]

    # Write the configuration to a JSON file
    with open("config.json", "w") as f:
        json.dump(config, f)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running chain in {dir}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", "chain_runner.py"])
