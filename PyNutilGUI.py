import tkinter 

from tkinter import *
from tkinter import ttk
import brainglobe_atlasapi
import PyNutil
from tkinter.filedialog import askopenfilename
from tkinter import colorchooser


#Basic GUI example   
root = Tk()
#root.geometry("300x300")
root.title("PyNutil")
root.wm_iconbitmap("Logo_PyNutil.ico")
#photo = tkinter.PhotoImage(file = 'Logo_PyNutil.ico')
#root.wm_iconphoto(False, photo)

arguments = {
   "registration_json":None,
   "object_colour":None
}

atlas = brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions()

selected_atlas = StringVar(value="Reference Atlas")

directory = ["select","select1", "select2"]
selected_directory = StringVar(value="directory")

colour = ["colour","black","red","blue","green"]
selected_colour = StringVar(value=colour[0])

def donothing():
   filewin = Toplevel(root)
   label = Label(filewin, text="Do nothing")
   label.pack()
   
def about_pynutil():
   filewin = Toplevel(root)
   label = Label(filewin, text="PyNutil is an application for brain-wide mapping using a reference brain atlas")
   label.pack()
   
def open_registration_json():
   value = askopenfilename()
   arguments["registration_json"] = value
   print(arguments)

def choose_colour():
   value = colorchooser.askcolor()
   arguments["object_colour"] = value
   print(list(value[0]))

def start_analysis():
   #your code here
   return

#Creating a menu
root.option_add('*tearOff', FALSE)
#win = Toplevel(root)
#menubar = Menu(win)
menubar = Menu(root)
#win['menu'] = menubar
root.config(menu=menubar)

#menubar = Menu(root)
menu_file = Menu(menubar)
menu_help = Menu(menubar)
menubar.add_cascade(menu=menu_file, label='File')
menubar.add_cascade(menu=menu_help, label='Help')

menu_file.add_command(label='New', command=donothing)
menu_file.add_command(label='Exit', command=root.quit)
menu_help.add_command(label='About PyNutil', command=about_pynutil)

#Creating a content frame"
mainframe = ttk.Frame(root, padding="12 12 12 12") # left top right bottom
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1) # column to expand if there is extra space
root.rowconfigure(0, weight=1) # row to expand if there is extra space

#Creating a content frame"
bottomframe = ttk.Frame(root, padding="12 12 12 12") # left top right bottom
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1) # column to expand if there is extra space
root.rowconfigure(0, weight=1) # row to expand if there is extra space

# Select reference atlas
ttk.Label(mainframe, text="Select reference atlas:", width=25).grid(column=1, row=1, sticky=W)
ttk.OptionMenu(mainframe, selected_atlas, "Reference Atlas", *atlas).grid(column=2, row=1, columnspan=2)
ttk.Button(mainframe, text="Help", width=8, command="buttonpressed").grid(column=4, row=1, sticky=W)

#Select registration JSON
ttk.Label(mainframe, text="Select registration JSON:", width=25).grid(column=1, row=2, sticky=W)
ttk.Button(mainframe, width=16, text="Browse...", command=open_registration_json).grid(column=2, row=2, sticky=W)
Text(mainframe, height = 1, width =40).grid(column=3, row=2, sticky= W)
ttk.Button(mainframe, text="Help", width=8, command="buttonpressed").grid(column=4, row=2, sticky=W)

#Select segmentation folder
ttk.Label(mainframe, text="Select segmentation folder:", width=25).grid(column=1, row=3, sticky=W)
ttk.Button(mainframe, width=16, text="Browse...", command="buttonpressed").grid(column=2, row=3, sticky=W)
Text(mainframe, height = 1, width =40).grid(column=3, row=3, sticky= W)
ttk.Button(mainframe, text="Help", width=8, command="buttonpressed").grid(column=4, row=3, sticky=W)

#Select object colour
ttk.Label(mainframe, text="Select object colour:", width=25).grid(column=1, row=4, sticky=W)
ttk.Button(mainframe, width=16, text="Colour", command=choose_colour).grid(column=2, row=4, sticky=W)
Text(mainframe, height = 1, width =40).grid(column=3, row=4, sticky= W)
ttk.Button(mainframe, text="Help", width=8, command="buttonpressed").grid(column=4, row=4, sticky=W)

#Select output directory
ttk.Label(mainframe, text="Select output directory:", width=25).grid(column=1, row=5, sticky=W)
ttk.Button(mainframe, width=16, text="Browse...", command="buttonpressed").grid(column=2, row=5, sticky=W)
Text(mainframe, height = 1, width =40).grid(column=3, row=5, sticky= W)
ttk.Button(mainframe, text="Help", width=8, command="buttonpressed").grid(column=4, row=5, sticky=W)

#Start analysis
ttk.Label(mainframe, text="Start analysis:", width=25).grid(column=1, row=6, sticky=W)
ttk.Button(mainframe, width = 52, text="Run", command="buttonpressed").grid(column= 3, row=6)
ttk.Button(mainframe, text="Docs", width=8, command="buttonpressed").grid(column=4, row=6, sticky=W)

# sunken frame around mainframe
"""
mainframe['borderwidth'] = 2
mainframe['relief'] = 'sunken'
"""

#button.configure()

root.mainloop()
