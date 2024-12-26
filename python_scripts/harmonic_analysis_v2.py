# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:31:51 2024

@author: csmall
"""

# Harmonic Tidal Analysis - Tests difference between pytides and utides

# Author: Chris Small, Coastal Engineer
# Email: csmall@eaest.com
# Organization: EA Engineering
# Date Created: 10/15/2024
# Date Revised: 10/16/2024
    
# This script pulls verified and predicted water levels from a NOAA station to 
# perform a harmonic analysis to make tidal predictions. Two different packages
#are used, pytides and utide, and a comparison of the results are made mack to 
# the NOAA data.

# The folder structure is auto generated from the script, user just has to save
# the script into the desired location and specify the path to this location. 

# Folder structure should be similar to below, but can vary.

# Folder structure when this code was written:

    # Parent folder: harmonic_analysis
        # Sub folder: scripts
            # haramonic_analysis_v1.py saved in this folder
        # Sub folder: analysis
            # Sub folder: figures
            # Sub folder: outputs
#%% Import required packages
import os
import noaa_coops as nc
from pytides.tide import Tide
import utide
import pandas as pd
import numpy as np
import matplotlib.image as image
from matplotlib import pyplot as plt, dates
#%% User Input Section

# Specify folder where script is saved
os.chdir(r'\Python\Templates\harmonic_analysis\scripts')

# Set working directory to be one folder up
os.chdir('../')

# Output directory for the figures
fig_path    = os.path.join(os.getcwd(),'analysis','figures')

# Check if path already exists, if not create the path
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    print(f'Created path: {fig_path}') # print out if path was created

# Output directory for csv files
csv_path    = os.path.join(os.getcwd(),'analysis','outputs')
    
# Check if path already exists, if not create the path
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
    print(f'Created path: {csv_path}') # print out if path was created

# Enter NOAA CO-OPS Station ID
st_ids       = [8452660] 
datum       = 'navd' # change as needed, NAVD is most common and preferred

# Define start and end dates (in UTC/GMT)
str_date    = '20220101 00:00' # set to match field data
end_date    = '20230101 00:00' # set to match field data

# Output names for the figures
fig_names   = ['prediction.png'] 

# Add output name to directory using list comprehension 
fig_out     = [os.path.join(fig_path, name) for name in fig_names]

# Output names for the csv files
csv_names   = ['noaa_data.csv'] 

# Add output name to directory using list comprehension 
csv_out     = [os.path.join(csv_path, name) for name in csv_names]

# Load graphics for inclusion on figures
# north arrow and EA logo
im = [image.imread(r'\Graphics\add_ons\north_arrow.png'),
                   image.imread(r'\Graphics\ea_logos\ea_logo_blue_name_to_side.png')]
#%% Define unit conversion functions

# Define conversion functions btw ft and m
def ft_to_m(x):
    return x*0.3048

def m_to_ft(x):
    return x*3.2808

# Define conversion functions btw knots and mph
def kn_to_ms(x):
    return x*0.514444

def ms_to_kn(x):
    return x*1.94384

# Define conversion functions btw psi and mb (note hPa and mb are equal)
def psi_to_mb(x):
    return x*68.9476

def mb_to_psi(x):
    return x*0.0145038
#%% Pull NOAA COOPs data

# Define function for extacting noaa station water level data
def fetch_station_wl_data(station_id, str_date, end_date, datum):
    
    st = nc.Station(station_id)
    st_data = st.get_data(
        begin_date=str_date, 
        end_date=end_date,
        product='hourly_height',
        datum=datum, 
        units='english',
        time_zone='gmt')
    
    # Reset index, drop nans, and remove any duplicates
    st_data = st_data.reset_index(drop=False).dropna().drop_duplicates(subset='t', keep='first')
        
    # Set datetime as index
    st_data = st_data.set_index('t')
    
    return st_data

# Fetch and process data for each station
st_data = {}
for st_id in st_ids:
    st_data[st_id] = fetch_station_wl_data(st_id, str_date, end_date, datum)

# Define function for extacting noaa station prediction data
def fetch_station_wl_pred_data(station_id, str_date, end_date, datum):
    
    st = nc.Station(station_id)
    st_data = st.get_data(
        begin_date=str_date, 
        end_date=end_date,
        product='predictions',
        datum=datum, 
        units='english',
        time_zone='gmt')
    
    # Reset index, drop nans, and remove any duplicates
    st_data = st_data.reset_index(drop=False).dropna().drop_duplicates(subset='t', keep='first')
        
    # Set datetime as index
    st_data = st_data.set_index('t')
    
    return st_data

# Fetch and process data for each station
st_pred = {}
for st_id in st_ids:
    st_pred[st_id] = fetch_station_wl_pred_data(st_id, str_date, end_date, datum)        
#%% Quick plot check

# Create figure
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15,10)) 

# water level verified
ax.plot(st_data[st_ids[0]].index, st_data[st_ids[0]]['v'], color = 'royalblue')
# water level predicted
ax.plot(st_pred[st_ids[0]].index, st_pred[st_ids[0]]['v'], color = 'green')
#%% Peform the harmonic analysis with pytide, following example here: https://ocefpaf.github.io/python4oceanographers/blog/2014/07/07/pytides/

# remove mean from the data
demeaned = st_data[st_ids[0]]['v'] - st_data[st_ids[0]]['v'].mean()

tide = Tide.decompose(demeaned, st_data[st_ids[0]].index)

constituent = [c.name for c in tide.model['constituent']]

df = pd.DataFrame(tide.model, index=constituent).drop('constituent', axis=1)

df.sort_values('amplitude', ascending=False).head(10)

print('Form number %s, the tide is %s.' % (tide.form_number()[0], tide.classify()))
#%% Test plot

dates = pd.date_range(start='2022-07-01', end='2022-07-30', freq='6T')

hours = np.cumsum(np.r_[0, [t.total_seconds() / 3600.0
                            for t in np.diff(dates.to_pydatetime())]])

times = Tide._times(dates[0], hours)

prediction = pd.DataFrame(tide.at(times) + st_data[st_ids[0]]['v'].mean(), index=dates)

# Create figure
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15,10)) 

# water level verified
ax.plot(st_data[st_ids[0]].index, st_data[st_ids[0]]['v'], color = 'royalblue')
# water level predicted
ax.plot(st_pred[st_ids[0]].index, st_pred[st_ids[0]]['v'], color = 'green')
ax.plot(prediction.index, prediction.values, color = 'red')
#%% Harmonic analysis with Utide, following example here: https://nbviewer.org/github/wesleybowman/UTide/blob/master/notebooks/utide_real_data_example.ipynb

# Remove mean from tidal signal
st_data[st_ids[0]]['demeaned'] = st_data[st_ids[0]]['v'] - st_data[st_ids[0]]['v'].mean()

# Solve for coefficients
coef = utide.solve(
    st_data[st_ids[0]].index,
    st_data[st_ids[0]]['demeaned'],
    lat=41,
    method="ols",
    conf_int="MC",
    verbose=False,
)

print(coef.keys())

# Reconstruct the tidal signal
tide = utide.reconstruct(st_data[st_ids[0]].index, coef, verbose=False)
print(tide.keys())

t = st_data[st_ids[0]].index.to_pydatetime()

fig, (ax0, ax1, ax2) = plt.subplots(figsize=(17, 5), nrows=3, sharey=True, sharex=True)

ax0.plot(t, st_data[st_ids[0]]['demeaned'], label="Observations", color="C0")
ax1.plot(st_pred[st_ids[0]].index, st_pred[st_ids[0]]['v'], label="Prediction", color="C1")
ax2.plot(t, st_data[st_ids[0]]['demeaned'] - tide.h, label="Residual", color="C2")
fig.legend(ncol=3, loc="upper center");
#%% Compare methods

# Create figure
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15,10)) 

# water level verified
ax.plot(st_data[st_ids[0]].index, st_data[st_ids[0]]['v'], color='royalblue',label='NOAA Verified')
# water level predicted
ax.plot(st_pred[st_ids[0]].index, st_pred[st_ids[0]]['v'], color='red',label='NOAA Prediction')
ax.plot(prediction.index, prediction.values, color = 'green',label='Pytides Prediction')
ax.plot(t, tide.h, color='magenta',label='Utide Prediction')

# fig.legend(ncol=4, loc="lower center")
ax.legend(ncol=4, loc='lower center', fontsize = 12, bbox_to_anchor=(0.5,-0.09))
# Set Titles
plt.suptitle('Harmonic Analysis Test - Newport, RI', fontsize=30, y=0.95)

# Set the second y axis to show metric values
secax0 = ax.secondary_yaxis('right', functions=(ft_to_m, m_to_ft))

# Label each y-axis
ax.set_ylabel('Water Level (ft, NAVD)', fontsize=18)
secax0.set_ylabel('Water Level (m, NAVD)', fontsize=18)

# Add grid lines
ax.grid(color='black', linestyle='--')
#%% Specify plot titles and legend items

# # Plot titles, grabs start/end year from dataframe and converts to string
# titles = [f"Cape Hatteras Environmental Conditions - {st_data[st_ids[0]]['month'][0]}/{st_data[st_ids[0]]['day'][0]}/{st_data[st_ids[0]]['year'][0]}" 
#               f" to {st_data[st_ids[0]]['month'][-1]}/{st_data[st_ids[0]]['day'][-1]}/{st_data[st_ids[0]]['year'][-1]}"]

# sub_titles = ['Air Pressure', 'Wind Speed and Direction', 'Water Level']

# # Axes and legend items
# axes_text = ['Air Pressure (psi)', 'Air Pressure (mb)', 'Wind Speed (kn)', 'Wind Speed (m/s)',
#              'Water Level (ft)','Water Level (m)']

# leg_text  = ['Cape Hatteras']
#%% Plot pressure, wind speed, and water level

# # # set date format for all following plots
# # date_form = dates.DateFormatter("%H:%M \n %m/%d")

# # # set default plot style
# # plt.style.use('ggplot')

# # Create figure
# fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,10)) 

# # plot the datasets

# # air Pressure 
# ax[0].plot(st_bp.index, st_bp['pres_psi'], color = 'red')
# # ax[0].set_ylim(ymax=max(st_bp['pres_psi']+0.5), ymin=min(st_bp['pres_psi'])-0.5)
# ax[0].set_xlim(xmax=st_bp.index[-1], xmin=st_bp.index[0])
# # ax[0].xaxis.set_major_formatter(date_form)

# # wind speed and wind direction 
# ax[1].plot(st_wd.index, st_wd['s'], '.g')
# ax[1].quiver(st_wd.index, st_wd['s'], st_wd['u'], st_wd['v'], 
#              color = 'black', width = 0.0007, headwidth = 6.5, scale = 750)
# # ax[1].plot(st_wd.index[::2], st_wd['s'][::2], '.g')
# # ax[1].quiver(st_wd.index[::2], st_wd['s'][::2], st_wd['u'][::2], st_wd['v'][::2], 
# #              color = 'black', width = 0.0007, headwidth = 6.5, scale = 750)
# ax[1].set_ylim(ymax=max(st_wd['s']+10), ymin=min(st_wd['s'])-5)
# ax[1].set_xlim(xmax=st_wd.index[-1], xmin=st_wd.index[0])
# # ax[1].xaxis.set_major_formatter(date_form)

# # water level
# ax[2].plot(st_data[st_ids[0]].index, st_data[st_ids[0]]['v'], color = 'royalblue')
# # ax[2].set_ylim(ymax=max(st_data[st_ids[0]]['v']+0.5), ymin=min(st_data[st_ids[0]]['v'])-0.5)
# ax[2].set_xlim(xmax=st_data[st_ids[0]].index[-1], xmin=st_data[st_ids[0]].index[0])
# # ax[2].xaxis.set_major_formatter(date_form)

# # Add the arrow as an inset axes
# ax_arrow = fig.add_axes([0.1, 0.52, 0.08, 0.08], anchor='SE')  # Adjust position and size as needed
# ax_arrow.imshow(im[0]) #, alpha=0.7)
# ax_arrow.axis('off')

# # Add the EA logo
# ax_logo = fig.add_axes([0.8, 0.025, 0.15, 0.15], anchor='SE')  # Adjust position and size as needed
# ax_logo.imshow(im[1]) #, alpha=0.7)
# ax_logo.axis('off')

# # Set Titles
# plt.suptitle(titles[0], fontsize=30, y=0.95)
# # ax.legend(ncol=len(leg_text), loc='lower center', fontsize = 12, bbox_to_anchor=(0.5,-0.03))

# # Subplot titles
# ax[0].set_title(sub_titles[0], fontsize = 18)
# ax[1].set_title(sub_titles[1], fontsize = 18)
# ax[2].set_title(sub_titles[2], fontsize = 18)

# # Set the second y axis to show metric values
# secax0 = ax[0].secondary_yaxis('right', functions=(psi_to_mb, mb_to_psi))
# secax1 = ax[1].secondary_yaxis('right', functions=(kn_to_ms, ms_to_kn))
# secax2 = ax[2].secondary_yaxis('right', functions=(ft_to_m, m_to_ft))

# # Label each y-axis
# ax[0].set_ylabel(axes_text[0], fontsize=18)
# secax0.set_ylabel(axes_text[1], fontsize=18)

# ax[1].set_ylabel(axes_text[2], fontsize=18)
# secax1.set_ylabel(axes_text[3], fontsize=18)

# ax[2].set_ylabel(axes_text[4], fontsize=18)
# secax2.set_ylabel(axes_text[5], fontsize=18)

# # Add grid lines for each subplot
# ax[0].grid(color='black', linestyle='--')
# ax[1].grid(color='black', linestyle='--')
# ax[2].grid(color='black', linestyle='--')

# # Rotate ticks to 45 degrees
# plt.xticks(rotation=45, fontsize=12)

# # Set xticks to XX day interval
# ax[0].xaxis.set_major_locator(dates.DayLocator(interval=5))

# # Save the figure
# plt.savefig(fig_out[0], bbox_inches='tight')
#%% Output water level data to csv for tidal datum calcs

# # Rename column headers
# st_wl = st_wl.rename(columns={'v':'Water Level (ft, LMSL)'})

# # Specify columns to ouput for the csv file
# header   = ['Water Level (ft, LMSL)']

# # Write field data to csv file
# field_data.to_csv(csv_out[0], columns=header)

# # Write noaa data to csv file
# st_wl.to_csv(csv_out[1], columns=header)