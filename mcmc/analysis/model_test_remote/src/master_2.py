import subprocess

#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#print("running first chain")
#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#subprocess.run("python simple.py", shell=True)

#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#print("running second chain")
#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#subprocess.run("python anchor.py", shell=True)

#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#print("running third chain")
#print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#subprocess.run("python curve.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running fourth chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python sigma.py", shell=True)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("running fifth chain")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
subprocess.run("python redshift.py", shell=True)
