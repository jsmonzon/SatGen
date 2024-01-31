import subprocess
import json

# Define your global variables here
config = {
    "location": "server",
    "a_stretch": 2.0,
    "nwalk": 100,
    "nstep": 2000,
    "ncores": 16,
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

mock_scripts = ["mock_1.py", "mock_2.py", "mock_3.py", "mock_4.py", "mock_5.py"]

for idx, mock_script in enumerate(mock_scripts, start=1):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"running model {idx}st mock")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    subprocess.run(["python", mock_script])
