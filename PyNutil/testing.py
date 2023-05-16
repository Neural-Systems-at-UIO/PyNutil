
import subprocess
import sys
import os
import glob

directory = ('../test')

"""
for filename in os.scandir(directory):
    if filename.is_file:
        print(filename.path)
"""        

directory = ('../test')

for filename in os.scandir(directory):
    if filename.path.endswith(".json") and filename.is_file:
        print(filename.path)


   

