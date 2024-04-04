import json
import re
import os

'''
Sharon Yates, 04.04.24.
This is a script for converting WALN and WWRP files from WebAlign and WebWarp to VisuAlign compatible JSON files.
To be used for testing purposes. 
'''

def waln_to_json(filename):
    with open(filename) as f:
        vafile = json.load(f)
    if filename.endswith(".waln") or filename.endswith("wwrp"):
        slices = vafile["sections"] # define slices as "section" in waln
        vafile["slices"] = slices 
        
        for slice in slices:
            print(slice) # this prints info from waln to screen.
            if "filename" in slice:
                base_name = os.path.basename(slice["filename"]).split('.')[0]
                new_filename = base_name + '.png'
                slice["filename"] = new_filename
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))
            if "ouv" in slice:
                slice["anchoring"] = slice["ouv"]        
        
        '''
        for slice in slices:
            print(slice) # this prints info from waln to screen.
            if "filename" in slice:
                name, old_extension = slice["filename"].rsplit('.',1)
                new_filename = name + '.png'
                slice["filename"] = new_filename
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))
            if "ouv" in slice:
                slice["anchoring"] = slice["ouv"]
        '''

        name = os.path.basename(filename)
        va_compat_file = {
            "name": name.replace(".waln",".json"),
            "target": vafile["atlas"] + '.cutlas',
            "target-resolution": [456, 528, 320],
            "slices": slices
        }
        # save with .json extension need to see if i can remove this
        with open(
            filename.replace(".waln", ".json").replace(".wwrp", ".json"), "w"
        ) as f:
            #json.dump(va_compat_file, f, indent=4) 
            json.dump(va_compat_file, f, indent=4)
            
        print("Waln or Wwrp converted successfully to JSON")

    else:
        pass
    
waln_to_json("PyNutil_test_2.waln")    
    

