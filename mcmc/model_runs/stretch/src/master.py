import subprocess

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running the 1.6 chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python test_16.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running the 1.8 chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python test_18.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running the 2.0 chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python test_20.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running the 2.2 chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python test_22.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running the 2.4 chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python test_24.py", shell=True)