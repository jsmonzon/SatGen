import subprocess
import json

# Define your global variables here
config = {
    "location": "local",
    "a_stretch": 2.0,
    "nwalk": 30,
    "nstep": 100,
    "ncores": 8,
    "min_mass": 6.5,
    "Ntree": 100,
    "N_corr": True,
    "p0_corr": True,
    "savefig": True,
    "reset": True,
}

# Write the configuration to a JSON file
with open("config.json", "w") as f:
    json.dump(config, f)

mock_scripts = ["mock_1.py", "mock_2.py", "mock_3.py"]

for idx, mock_script in enumerate(mock_scripts, start=1):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running model {idx}st mock")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", mock_script])
