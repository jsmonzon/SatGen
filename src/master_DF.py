import subprocess

scripts = ["jsm_SubEvo_DF_1.py", "jsm_SubEvo_DF_2.py", "jsm_SubEvo_DF_3.py"]

for script in scripts:

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    subprocess.run(["python", script])
