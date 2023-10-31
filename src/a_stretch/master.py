import subprocess

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running first chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python run1.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running second chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python run2.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running third chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python run3.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running fourth chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python run4.py", shell=True)