   
"""This is Gergely Csusc's code from VisuAlign MediaWiki NITRC page,
this was a test, not used elsewhere in PyNutil"""   
   
   import sys
   for arg in sys.argv:
       if arg.startswith("flat="):
           flatfile=arg[len("flat="):]
       if arg.startswith("label="):
           labelfile=arg[len("label="):]
       if arg.startswith("json="):
           jsonfile=arg[len("json="):]
       if arg.startswith("output="):
           outfile=arg[len("output="):]
   if "flatfile" not in vars():
       print("flat=<some .flat file> parameter missing")
       sys.exit()
   if "outfile" not in vars():
       print("output=<new .png file> parameter missing")
       sys.exit()
   
   palette=False
   if "labelfile" in vars():
       import re
       palette=[]
       with open(labelfile) as f:
           for line in f:
               lbl=re.match(r'\s*\d+\s+(\d+)\s+(\d+)\s+(\d+)',line)
               if lbl:
                   palette.append((int(lbl[1]),int(lbl[2]),int(lbl[3])))
       print(f"{len(palette)} labels parsed")
   elif "jsonfile" in vars():
       import json
       with open(jsonfile) as f:
           palette=[(i["red"],i["green"],i["blue"]) for i in json.load(f)]
       print(f"{len(palette)} labels loaded")
   
   import struct
   with open(flatfile,"rb") as f:
       b,w,h=struct.unpack(">BII",f.read(9))
       data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
   print(f"{b} bytes per pixel, {w} x {h} resolution")
   
   import PIL.Image
   image=PIL.Image.new("RGB" if palette else "L",(w,h))
   for y in range(h):
       for x in range(w):
           image.putpixel((x,y),palette[data[x+y*w]] if palette else data[x+y*w] & 255)
   image.save(outfile,"PNG")