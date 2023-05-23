
import subprocess
import sys
import os
import glob

dir = ('../test_data/ttA_2877_NOP_atlasmaps')     
"""fetch file names in a directory"""

def FilesinDirectory(directory):
    for file in os.scandir(directory):
        if file.path.endswith(".flat") and file.is_file:
            #print(filename.path)
            #newfilename, file_ext = os.path.splitext(filename)
            #print(newfilename)
            filename = os.path.basename(file)
            newfilename, file_ext = os.path.splitext(filename)
            return newfilename

#Question: how to return multiple file names into a list?


files = []
newfiles = files.append(FilesinDirectory('../test_data/ttA_2877_NOP_atlasmaps'))
print(files)

        



import os

# file name with extension
file_name = os.path.basename('../test_data/ttA_2877_NOP_atlasmaps')

print(file_name)

# file name without extension
print(os.path.splitext(file_name)[0])


   

