# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:58:56 2025
@author: mmchugh

Tidal Datum Plot

Author: Maya McHugh, Coastal Engineer
Email: mmchugh@eaest.com
Organization: EA Engineering
Date Created: 02/28/2025

This script takes user or auto input tidal datums, which can be referenced from 
the NOAA Tides and Currents site: https://tidesandcurrents.noaa.gov/map/index.html
and outputs a tidal datum figure. 

The folder structure is auto generated from the script, user just has to save
the script into the desired location. 
"""
#%% Import required packages
import os
import sys
import noaa_coops as nc
import webbrowser as wb
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import matplotlib.image as image
#%% User Input Section

# Need to define paths different, compiling exe messes with it

def resource_path(relative_path):
    """ Get the path to a resource, works for dev and PyInstaller bundle. """
    try:
        # PyInstaller creates a _pyinstaller directory in the temporary folder
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# Define project directory
# project_dir     = os.path.abspath('.')

# Define project directory: Use the script's or executable's directory
if getattr(sys, 'frozen', False):  # If running as a PyInstaller executable
    project_dir = os.path.dirname(sys.executable)
else:
    project_dir = os.path.dirname(os.path.abspath(__file__))

# Specify output directory for the figures
figure_dir  = os.path.join(project_dir, 'figures')

# Check if paths already exists, if not create the path
# os.makedirs(project_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# Print out directory locations
print(f"Project Directory: {project_dir}")
print(f"Figure Directory: {figure_dir}")
#%% Setup prompt for user to enter NOAA CO-OPS Station ID and select datum method

class StationDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        self.st_id = None
        self.manual_entry = False
        super().__init__(parent, title)

    def body(self, master):
        tk.Label(master, text="Enter NOAA CO-OPS Station ID (e.g., 8656483):").grid(row=0, column=0, sticky="w")
        self.st_id_entry = tk.Entry(master)
        self.st_id_entry.grid(row=0, column=1)

        self.manual_var = tk.IntVar()
        self.manual_check = tk.Checkbutton(master, text="Enter tidal datum values manually? (Leave unchecked for auto extract)", variable=self.manual_var)
        self.manual_check.grid(row=1, columnspan=2, sticky="w")

        return self.st_id_entry  

    def apply(self):
        self.st_id = self.st_id_entry.get().strip()
        self.manual_entry = bool(self.manual_var.get())
        # self.manual_entry = self.manual_var.get() == 1

# Create a hidden root window for the pop-up
root = tk.Tk()
root.withdraw()

# Show the dialog box
dialog = StationDialog(root, "Station ID & Tidal Datum Method")

# Extract results
st_id           = dialog.st_id
manual_entry    = dialog.manual_entry

# Handle empty station ID case
if not st_id:
    tk.messagebox.showerror("Error", "User cancelled or entered invalid input, close window and retry.")
    root.destroy()
    exit()

root.destroy()  # Close window
#%% Pull station info and set figure directory

datum       = 'NAVD88' # change as needed, NAVD is most common
url         = f"https://tidesandcurrents.noaa.gov/datums.html?datum={datum}&units=0&epoch=0&id={st_id}"

# Add logic to alert user if station data pull fails
try: 
    st          = nc.Station(st_id)
    
except Exception as e:
    print(f"Error retrieving NOAA station info: {e}, close window and retry")
    #sys.exit(1)  # Exit script on failure
    
# Output names for the figures
fig_name   = f"noaa_st_{st_id}_tidal_datums.png"

# Add output name to directory using list comprehension 
fig_out     = os.path.join(figure_dir, fig_name)
#%% Setup logic so that datums can be enter manually or taken from the api

# Manual entry datum logic
if manual_entry == True:
    
    # Setup logic to handle if url values
    try:
        
        # Open NOAA Tides and Currents Page and enter values
        wb.open(url)
    
    # Error handling 
    except Exception as e:
        print(f"Error connecting to {url}, NAVD88 likely not valid option but can still enter values manually: {e}")

    class MultiFloatDialog(simpledialog.Dialog):
        def __init__(self, parent, title=None, prompts=None):
            self.prompts = prompts or ["Value 1", "Value 2"]
            self.values = [None] * len(self.prompts)
            super().__init__(parent, title)

        def body(self, master):
            self.entries = []
            for i, prompt in enumerate(self.prompts):
                tk.Label(master, text=prompt).grid(row=i)
                entry = tk.Entry(master)
                entry.grid(row=i, column=1)
                self.entries.append(entry)
            return self.entries[0] # initial focus

        def apply(self):
            for i, entry in enumerate(self.entries):
                try:
                    self.values[i] = float(entry.get())
                except ValueError:
                   tk.messagebox.showerror("Error", f"Invalid input for {self.prompts[i]}. Please enter a number, close window and retry.")
                   return  # Cancel apply if there's an error
            self.result = self.values

    if __name__ == '__main__':
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        prompts = ["MHHW","MHW", "MSL","MLW","MLLW"]
        dialog = MultiFloatDialog(root, "Enter Dimensions", prompts)

        if dialog.result:
            t_datums = [MHHW, MHW, MSL, MLW, MLLW] = dialog.result
        else:
            print("User cancelled or entered invalid input, close window and retry.")

        root.destroy() 

# Auto extract datum logic        
else:
    # Using tidal datum values from API, come referenced to STND
    datums_stnd = {
        'STND': st.datums['datums'][0]['value'],
        'MHHW': st.datums['datums'][1]['value'],
        'MHW': st.datums['datums'][2]['value'],
        'MSL': st.datums['datums'][5]['value'],
        'MLW': st.datums['datums'][6]['value'],
        'MLLW': st.datums['datums'][7]['value'],
        'NAVD88': st.datums['datums'][14]['value'],
    }
    
    # Adjust datums to NAVD88
    stnd_to_navd = datums_stnd['STND'] - datums_stnd['NAVD88']
    
    # Tidal datums relative to NAVD88    
    datums_navd = {key: round(value + stnd_to_navd, 2) for key, value in datums_stnd.items()}
    
    # Carrying over so plotting can stay the same
    prompts = ['MHHW','MHW', 'MSL','MLW','MLLW']
    
    # Setup variable for plotting and round numbers    
    t_datums = [datums_navd[key] for key in prompts]
    
    # Pull out values individually
    MHHW, MHW, MSL, MLW, MLLW = t_datums
#%% Plot Tidal Datum Figure

# Logic for if exe or script
# Load the background image and EA logo
if getattr(sys, 'frozen', False):  # If running as a PyInstaller executable
    # PyInstaller creates a _pyinstaller directory in the temporary folder
    base_path   = sys._MEIPASS
    bg_image    = plt.imread(os.path.join(base_path, 'graphics', 'background.jpg'))
    ea_logo     = image.imread(os.path.join(base_path, 'graphics', 'ea_logo.png'))
    
else:
    bg_image    = plt.imread(os.path.join(project_dir, 'graphics', 'background.jpg'))
    ea_logo     = image.imread(os.path.join(project_dir, 'graphics', 'ea_logo.png'))
    
# Define the range for the lines 
gap = (MHW-MLW)/4 # space above and below datums
mingap = min(MHHW-MHW,MLW-MLLW)/2 
ymin = round((MLLW-gap),0)
ymax = round((MHHW+gap),0)
axlen = ymax-ymin
# Background scale centered on MSL
bg_min = MSL-(axlen/2)
bg_max = MSL+(axlen/2)

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(6,8))

# Plotting the horizontal lines using axhline
for y in t_datums:
    plt.axhline(y=y, xmin=0.5, xmax=1, color='k', linestyle='--')

# Datum line labels
for i, j in enumerate(t_datums):
    text = prompts[i] + ': ' + str(t_datums[i])
    plt.text(axlen/2, j, text, fontsize=10, ha='right', va='center')

# Text for title
fig.text(0.14, 0.856, f"Tidal Datums for {st.name}, {st.state} {st_id}",  fontsize=10)

# Text for y-axis
plt.ylabel('Elevation in feet relative to NAVD88')

# Hide x-axis ticks
ax.set_xticks([])

# Set plot limits to match the data
# ax.set_ylim(axmin, axmax)

# Set the background image
ax.imshow(bg_image, extent=[0, axlen, bg_min, bg_max], aspect='auto', zorder=-1)

# Add the EA logo
ax_logo = fig.add_axes([0.15, 0.12, 0.3, 0.3], anchor='SW')  # Adjust position and size as needed
ax_logo.imshow(ea_logo) #, alpha=0.7)
ax_logo.axis('off')

# Add disclaimer text at the bottom of the figure
fig.text(0.52, 0.12, ('Disclaimer: Any conclusions drawn from\n the analysis of this ' 
         'information are not\n the responsibility of EA or its partners.'), fontsize=8)

# Save the figure
plt.savefig(fig_out, bbox_inches='tight')