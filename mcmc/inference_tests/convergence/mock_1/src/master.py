import subprocess
import json

mock_dirs = ["../Nhost_100/"] #, "../Nhost_1000/"] #, "../Nhost_10000/"]

config = {
    "location": "server",
    "a_stretch": 1.2,
    "nwalk": 15,
    "nstep": 35000,
    "ncores": 16,
    "min_mass": 6.5,
    "max_mass":11.5,
    "Nsamp": 1,
    "max_N": 1000,
    "init_gauss": 1e-4,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": False,
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