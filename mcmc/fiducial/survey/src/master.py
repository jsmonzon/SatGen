import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "a_stretch": 2.0,
    "nwalk": 15,
    "nstep": 5000,
    "ncores": 16,
    "min_mass": 6.5,
    "max_N": 2500,
    "Nsamp": 1,
    "init_gauss": 1e-2,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

mock_scripts = ["mock_55.py", "mock_75.py", "mock_300.py", "mock_500.py", "mock_1000.py"]

for idx, mock_script in enumerate(mock_scripts, start=1):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running {idx} mock")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", mock_script])
