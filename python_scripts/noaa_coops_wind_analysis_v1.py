# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:36:47 2023

@author: csmall
"""

# NOAA CO-OPS Wind Analysis Example

# Author: Chris Small, Coastal Engineer
# Email: csmall@eaest.com
# Organization: EA Engineering
# Date Created: 04/06/2023
# Date Revised: 11/13/2023

# This script pulls data directly from the NOAA CO-OPS Ocean City, MD Station,
# ID: 8570283. Script plots a full period rose, seasonal wave roses subplot, 
# full timeseries subplot with barometric pressure, wind magnitude, and 
# direction, full period box plot, monthly means box plot, and monthly max
# boxplot.

# The folder structure is auto generated from the script, user just has to save
# the script into the desired location and specify the path to this location. 
# The code will then move up one folder and create an "Analysis" folder with 
# additional subfolders for "Wind" and "Figures". The output plots will be 
# saved in the subfolder called "Figures".

# Update path to script, line 57:
    # (os.chdir(r'C:\Users\csmall\Documents\User_Documents\Python\Templates\noaa_coops_example\Scripts')

# Folder structure should be similar to below, but can vary.

# Folder structure when this code was written:

    # Parent folder: Assateague
        # Sub folder: Scripts
            # assateague_windanalysis_v2.py saved in this folder
        # Sub folder: Analysis
            #Sub folder: Wind
                # Sub folder: Figures
                    # Folder where output figures are saved
                    
# Note you may need to adjust the bins_range, st_val, it_val, and rmax values
# to fix formatting issues for polar scaling.
#%% Import required packages
import os
import noaa_coops as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import matplotlib.legend_handler
import seaborn as sns
#%% User Input Section

# Specify folder where script is saved
os.chdir(r'C:\Users\csmall\Documents\User_Documents\Python\Templates\noaa_coops_example\Scripts')

# Set working directory to be one folder up
os.chdir('../')

# Output directory for the figure
fig_path    = os.path.join(os.getcwd(),'Analysis','Wind','Figures')
    
# Check if path already exists, if not create the path
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    print(f'Created path: {fig_path}') # print out if path was created

# Enter NOAA CO-OPS Station ID
st_id       = 8570283

# Define start and end dates for data/plots (in UTC/GMT)
str_date    = '20210101 00:00'
end_date    = '20230331 00:00'

# Output name for the figure
fig_names   = ('Assateague_WindRose_FullPeriod_v1.png','Assateague_WindRose_Seasonal_v1.png', 
               'Assateague_Wind_Timeseries_v1.png','Assateague_Boxplot_FullPeriod_v1.png', 
               'Assateague_Boxplot_MonthlyMeans_v1.png','Assateague_Boxplot_MonthlyMaxes_v1.png') 

# Add output name to directory using list comprehension 
fig_out     = [os.path.join(fig_path, name) for name in fig_names]
#%% Pull NOAA CO-OPs data

# List of product names
products    = ['wind', 'air_pressure']

# Set NOAA CO-OPs Station
st = nc.Station(st_id)

# Empty list to store data
data        = []

# Loop through product names and pull data
for product in products:
    # Pull data for the current product
    df = st.get_data(begin_date=str_date, 
                     end_date=end_date,
                     product=product,
                     interval='h',
                     units='english',
                     time_zone='gmt')
    
    # Reset index and drop any rows with NaN values
    df = df.reset_index(drop=False).dropna()
    df = df.set_index('t')
    # Append the cleaned data to the list
    data.append(df)

# Unpack dataframes list into separate variables
st_wd, st_bp    = data

# Drop off zero wind data
st_wd           = st_wd.query('s != 0')
#%% Define functions for converting between US and metric units

# Define conversion functions btw knots and m/s
def kn_to_ms(x):
    return x*0.5144

def ms_to_kn(x):
    return x*1.9438

# Define conversion functions btw knots and mph
def kn_to_mph(x):
    return x*0.868976

def mph_to_kn(x):
    return x*1.15078

# Define conversion functions btw psi and mb (note hPa and mb are equal)
def psi_to_mb(x):
    return x*68.9476

def mb_to_psi(x):
    return x*0.0145038
#%% Break wind data out into months/seasons for further analysis

# Disable false positive warning message that appears
pd.options.mode.chained_assignment = None  # default='warn'

# Split date/time into year, month, and day
st_wd['year']   = st_wd.index.year
st_wd['month']  = st_wd.index.month
st_wd['day']    = st_wd.index.day

# Define variables for seasons
winter          = [1,2,12]
spring          = [3,4,5]
summer          = [6,7,8]
autumn          = [9,10,11]

# Pull out data based on seasons
st_wd_winter    = st_wd.loc[st_wd['month'].isin(winter)]
st_wd_spring    = st_wd.loc[st_wd['month'].isin(spring)]
st_wd_summer    = st_wd.loc[st_wd['month'].isin(summer)]
st_wd_autumn    = st_wd.loc[st_wd['month'].isin(autumn)] 

# Put all the seasonal data together
st_wd_seasonal  = [st_wd_summer, st_wd_autumn, st_wd_winter, st_wd_spring]

# Pull out monthly means 
st_mn           = st_wd.groupby(['year','month'])['s'].mean()
st_mn           = pd.DataFrame(st_mn).reset_index()
st_mn.columns   = ['year','month','wind_speed_mean']

# Pull out monthly maxes
st_mx           = st_wd.groupby(['year','month'])['s'].max()
st_mx           = pd.DataFrame(st_mx).reset_index()
st_mx.columns   = ['year','month','wind_speed_max']
#%% Specify plot titles and legend items

# Plot titles, grabs save point number and start/end year from dataframe and converts to string
titles = [f"{st.id} - {st.name}, {st.state} - Full Period, {st_wd['year'][0]} to {st_wd['year'][-1]}",
          'Barometric Pressure', 'Wind Speed', 'Wind Direction',
          f"{st.id} - {st.name}, {st.state} - {st_wd['year'][0]} to {st_wd['year'][-1]} - Monthly Means",
          f"{st.id} - {st.name}, {st.state} - {st_wd['year'][0]} to {st_wd['year'][-1]} - Monthly Maxes"]

# Specify season names for subplot titles
seasons = ['Summer (Jun, Jul, Aug)', 'Fall (Sep, Oct, Nov)', 
           'Winter (Dec, Jan, Feb)', 'Spring (Mar, Apr, May)']

# Legend items
leg_text = ['Wind Speed (kn)','Baro','Gusts','Sustained']
#%% Full period rose plot
    
# Set dark grid background for all plots
# sns.set_theme(style='darkgrid')

# Set bins for the data
bins_range  = [0, 7, 14, 21] # values in kn, bins for wind
# bins_range  = [0, 10, 21, 33] # values in kn, bins for wind

# Set colormap for rose plot
c_map       = plt.get_cmap('plasma')

# Define variable to adjust max polar range
pol_adj     = 6 # Vary this number to get desired look

# Define start and interval values for the rose range
st_val      = 3 # value of the inner most polar circle
it_val      = 3 # interval value, ie 6, 12, 18...

# Plot rose
ax = WindroseAxes.from_ax()
# Grab direction and magnitude data
ax.bar(st_wd['d'], st_wd['s'], 
       normed=True, opening=0.8, edgecolor='white', bins=bins_range, cmap=c_map,
       calm_limit=0)

# Set range for the rose (% values), need to be careful with this step
# NEED TO CHECK THIS RANGE WORKS EACH TIME, if not vary 'pol_adj', 'st_val',
# and/or 'it_val' variables
ranges = np.arange(st_val, round(ax.rmax)+pol_adj, step=it_val) 

# Create variable for yticks based on ranges defined above
y_text = [f'{ranges[ind]}%' for ind in np.ndindex(ranges.shape)] # adds % sign to range values

# Finish rest of rose formatting    
ax.set_yticks(ranges) # set range for polars
ax.set_yticklabels(ranges) # set labels for polars
ax.set_yticklabels(y_text) # add % sign to labels
# rlabels = ax.get_ymajorticklabels()
# for label in rlabels:
#     label.set_color('white')
ax.set_xticklabels(['E', 'NE','N', 'NW', 'W', 'SW', 'S', 'SE']) # set labels
leg = ax.legend(fontsize = 13, loc ='lower right') # changes font size
leg.set_title(leg_text[0], prop = {'size':13}) # changes legend size
plt.title(titles[0], fontsize=17, y=1.06) #y=1.08 raises the title

# Save the figure
plt.savefig(fig_out[0])
#%% Seasonal wind rose subplot

# Set bins for the data
bins_range  = [0, 7, 14, 21] # values in kn, bins for wind

# Define start and interval values for the rose range
st_val      = 4 # value of the inner most polar circle
it_val      = 4 # interval value, ie 6, 12, 18...

nrows, ncols = 2, 2
fig = plt.figure(figsize=(15, 15))
# Set overall title
plt.suptitle(titles[0], fontsize=25, y=0.95)

# Loop to create seasonal rose figure
for i, ind in enumerate(st_wd_seasonal):
    
    # Lines that create and format roses
    ax      = fig.add_subplot(nrows, ncols, i + 1, projection='windrose', rmax=20)
    ax.bar(st_wd_seasonal[i]['d'], st_wd_seasonal[i]['s'], normed=True, opening=0.8,
           edgecolor='white', bins=bins_range, cmap=c_map, calm_limit=0)
   
    # Create range for polar ticks (+1 is so rmax value is included)
    ranges  = np.arange(st_val, ax.rmax+1, step=it_val) 
    
    # Create variable for yticks based on ranges defined above
    y_text  = [f'{ranges[ind]}%' for ind in np.ndindex(ranges.shape)] # adds % sign to range values
    
    # Finish rest of rose formatting    
    ax.set_yticks(ranges) # set range for polars
    ax.set_yticklabels(ranges) # set labels for polars
    ax.set_yticklabels(y_text) # add % sign to labels
    ax.set_xticklabels(['E', 'NE','N', 'NW', 'W', 'SW', 'S', 'SE']) # set labels
    plt.title(seasons[i], fontsize=17, y=1.06) #y=1.06 raises the title

# Set legend outside of loop to only create one
leg = ax.legend(fontsize = 13, loc='lower right') # changes font size
leg.set_title(leg_text[0], prop = {'size':13}) # changes legend size  
  
# Save the figure
plt.savefig(fig_out[1], bbox_inches='tight')
#%% Plot barometric pressure, wind speed, and wind direction

# Crete figure and specify figure size
f, ax = plt.subplots(3, 1, sharex=True, figsize=(15,12)) 

# Plot the datasets
# Plot data for first subplot
l1, = ax[0].plot(st_bp.index, mb_to_psi(st_bp['v']), 'go', 
                  markersize=1, label=leg_text[1], zorder=10)

# Plot data for second subplot
l2, = ax[1].plot(st_wd.index, st_wd['g'], 'b', linewidth=1, 
            label=leg_text[1], zorder=10)
l3, = ax[1].plot(st_wd.index, st_wd['s'], 'r', linewidth=1, 
            label=leg_text[1], zorder=10)
    
# Plot data for the third subplot
l4, = ax[2].plot(st_wd.index, st_wd['d'], 'bo', 
                  markersize=1, label=leg_text[1], zorder=10)
l5, = ax[2].plot(st_wd.index, st_wd['d'], 'ro', 
                  markersize=1, label=leg_text[1], zorder=10)

# Set overall titles
plt.suptitle(titles[0], fontsize=25, y=0.95)

# Subplot titles
ax[0].set_title(titles[1], fontsize = 18)
ax[1].set_title(titles[2], fontsize = 18)
ax[2].set_title(titles[3], fontsize = 18)

# Set the second y axis to show metric values
secax0 = ax[0].secondary_yaxis('right', functions=(psi_to_mb, mb_to_psi))
secax1 = ax[1].secondary_yaxis('right', functions=(kn_to_ms, ms_to_kn))
secax2 = ax[2].secondary_yaxis('right')

# Label each y-axis
secax0.set_ylabel('Pressure (mb)', fontsize=18)
ax[0].set_ylabel('Pressure (psi)', fontsize=18)

secax1.set_ylabel('Wind Speed (m/s)', fontsize=18)
ax[1].set_ylabel('Wind Speed (kn)', fontsize=18)

secax2.set_ylabel('Direction ($^\circ$)', fontsize=18)
ax[2].set_ylabel('Direction ($^\circ$)', fontsize=18)

# Add grid lines for each subplot
ax[0].grid(color='black', linestyle='--')
ax[1].grid(color='black', linestyle='--')
ax[2].grid(color='black', linestyle='--')

# Rotate ticks to 45 degrees
plt.xticks(rotation=45)
        
# Set Global Legend
lines    = []
labels   = []

for x in f.axes:
    axLine, axLabel = x.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

# Grab unique lines, bit manual, definitely a better way to do this
leg_lines = (lines[0], lines[1], lines[2])

# Grab plot handles for tuple
handles = [l1, (l2,l4), (l3,l5)]

f.legend(handles = handles, labels=[leg_text[1], leg_text[2], leg_text[3]], 
          loc='lower center', ncol=len(leg_lines), fontsize = 12, 
          handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(None)})

# Workaround for setting common axis titles, add a big axis, hide frame
f.add_subplot(111, frameon=False)

# Hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# Label x axis
plt.xlabel('Year', fontsize=18, labelpad=20)

# Save the figure
plt.savefig(fig_out[2], bbox_inches='tight')
#%% Box plot - full period

# Parameters for controling boxplot look
red_circ = dict(markerfacecolor='red', marker='o')
mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='black')

# Create figure and specify subplot orientation (1 rows, 1 column), shared x-axis, and figure size
f, ax = plt.subplots(1, 1, sharex=True, figsize=(15,6)) 

# Create the boxplot
sns.boxplot(x='month',y='s', data=st_wd, ax=ax, showmeans=True, 
            meanprops=mean_shape, palette = 'Blues', flierprops=red_circ)

# Show underlying data points
# sns.stripplot(x='month',y='s', data=st_wd, ax=ax)

# Set the ticks for months on the x-axis
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],['Jan','Feb','Mar','Apr','May','June','July',
                                         'Aug','Sep','Oct','Nov','Dec'])

# Set title
plt.suptitle(titles[0], fontsize=25, y=0.95)
# ax.legend(loc='lower left', fontsize = 12)

# Add second y-axis with metric values
secax = ax.secondary_yaxis('right', functions=(kn_to_ms, ms_to_kn))

# Label x-axis
plt.xlabel('Month', fontsize=18)

# Label each y-axis
secax.set_ylabel('Wind Speed (m/s)', fontsize=18)
ax.set_ylabel('Wind Speed (kn)', fontsize=18)

# Add grid lines 
ax.set_axisbelow(True) # sets lines below data
ax.grid(color='black', linestyle='--')

# Save the figure
plt.savefig(fig_out[3], bbox_inches='tight')
#%% Box plot - monthly means

# Parameters for controling boxplot look
red_circ = dict(markerfacecolor='red', marker='o')
mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='black')

# Create figure and specify subplot orientation (1 rows, 1 column), shared x-axis, and figure size
f, ax = plt.subplots(1, 1, sharex=True, figsize=(15,6)) 

# Create the boxplot
sns.boxplot(x='month',y='wind_speed_mean', data=st_mn, ax=ax, showmeans=True, 
            meanprops=mean_shape, palette = 'Blues', flierprops=red_circ)

# Show underlying data points
# sns.stripplot(x='month',y='wind_speed_mean', data=st_mn, ax=ax)

# Set the ticks for months on the x-axis
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],['Jan','Feb','Mar','Apr','May','June','July',
                                         'Aug','Sep','Oct','Nov','Dec'])

# Set title
plt.suptitle(titles[4], fontsize=25, y=0.95)
# ax.legend(loc='lower left', fontsize = 12)

# Add second y-axis with metric values
secax = ax.secondary_yaxis('right', functions=(kn_to_ms, ms_to_kn))

# Label x-axis
plt.xlabel('Month', fontsize=18)

# Label each y-axis
secax.set_ylabel('Wind Speed (m/s)', fontsize=18)
ax.set_ylabel('Wind Speed (kn)', fontsize=18)

# Add grid lines 
ax.set_axisbelow(True) # sets line below data
ax.grid(color='black', linestyle='--')

# Save the figure
plt.savefig(fig_out[4], bbox_inches='tight')
#%% Box plot - monthly maxes

# Parameters for controling boxplot look
red_circ = dict(markerfacecolor='red', marker='o')
mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='black')

# Create figure and specify subplot orientation (1 rows, 1 column), shared x-axis, and figure size
f, ax = plt.subplots(1, 1, sharex=True, figsize=(15,6)) 

# Create the boxplot
sns.boxplot(x='month',y='wind_speed_max', data=st_mx, ax=ax, showmeans=True, 
            meanprops=mean_shape, palette = 'Blues', flierprops=red_circ)

# Show underlying data points
# sns.stripplot(x='month',y='wind_speed_max', data=st_mx, ax=ax)

# Set the ticks for months on the x-axis
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],['Jan','Feb','Mar','Apr','May','June','July',
                                         'Aug','Sep','Oct','Nov','Dec'])

# Set title
plt.suptitle(titles[5], fontsize=25, y=0.95)
# ax.legend(loc='lower left', fontsize = 12)

# Add second y-axis with metric values
secax = ax.secondary_yaxis('right', functions=(kn_to_ms, ms_to_kn))

# Label x-axis
plt.xlabel('Month', fontsize=18)

# Label each y-axis
secax.set_ylabel('Wind Speed (m/s)', fontsize=18)
ax.set_ylabel('Wind Speed (kn)', fontsize=18)

# Add grid lines 
ax.set_axisbelow(True) # sets lines below data
ax.grid(color='black', linestyle='--')

# Save the figure
plt.savefig(fig_out[5], bbox_inches='tight')