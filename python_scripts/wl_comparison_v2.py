# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:01:49 2024

@author: csmall
"""

# Water Level Comparison Template

# Author: Chris Small, Coastal Engineer
# Email: csmall@eaest.com
# Organization: EA Engineering
# Date Created: 04/18/2024
# Date Revised: 

# Currently setup using field data from Cherry Point as an example 
    
# This script perfroms a water level comparison between field data and a NOAA
# tidal station. Field data is expected to be a csv and headers must match.

# Expects field data to be datetime in GMT/UTC and water level to be feet
# referenced to NAVD88.

# The folder structure is auto generated from the script, user just has to save
# the script into the desired location and specify the path to this location. 

# Folder structure should be similar to below, but can vary.

# Folder structure when this code was written:

    # Parent folder: water_level_comparison
        # Sub folder: scripts
            # wl_comparison_v1.py saved in this folder
        # Sub folder: analysis
            # Sub folder: data
                # field_data.csv
                    # Header 1 - Datetime GMT
                    # Header 2 - Water Level (ft, NAVD)
#%% Import required packages
import os
import noaa_coops as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%% User Input Section

# Specify folder where script is saved
os.chdir(r'C:\Users\csmall\Documents\User_Documents\Python\Templates\water_level_comparison\scripts')

# Set working directory to be one folder up
os.chdir('../')

# Set file ID and path to field data file
fid         = [os.path.join(os.getcwd(),'analysis','data',
                            'HydroVu_Troll-927471_2024-03-26_04-32-00_Export.csv')] 

# Number of rows to skip at start of csv (up to the header line)
skip        = 8

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
st_id       = 8655133 # Oriental, Neuse River 
datum       = 'navd' # change as needed, NAVD is most common

# Define start and end dates (in UTC/GMT)
str_date    = '20240326 00:00' # set to match field data
end_date    = '20240424 14:00' # set to match field data

# Convert HST to UTC/GMT for comparison
# hr = 10 # measured data was 10 hours behind GMT

# Output names for the figures
fig_names   = ['water_level_comparison_v1.png'] 

# Add output name to directory using list comprehension 
fig_out     = [os.path.join(fig_path, name) for name in fig_names]

# Output names for the csv files
csv_names   = ['field_data.csv', 'noaa_data.csv'] 

# Add output name to directory using list comprehension 
csv_out     = [os.path.join(csv_path, name) for name in csv_names]
#%% Define unit conversion functions

# Define conversion functions btw ft and m
def ft_to_m(x):
    return x*0.3048

def m_to_ft(x):
    return x*3.2808
#%% Load in field data from csv files

# Import data from csv file
field_data                  = pd.read_csv(fid[0], skiprows=skip)

# convert date/time column to datetime for plotting
field_data['datetime_gmt']  = pd.to_datetime(field_data['Datetime GMT'])

# Create date/time adjusted to UTC/GMT
# field_data['datetime_gmt']  = field_data['datetime'] + pd.Timedelta(hours=hr)

# # Remove data before and after certain dates
# field_data = field_data[~(field_data['datetime_gmt'] < str_date)]
# field_data = field_data[~(field_data['datetime_gmt'] > end_date)]

# Reset index and drop nans
field_data                  = field_data.reset_index(drop=False)
field_data                  = field_data.dropna()

# Set datetime as index
field_data                  = field_data.set_index('datetime_gmt')

# Split date/time into year, month, and day
field_data['year']          = field_data.index.year
field_data['month']         = field_data.index.month
field_data['day']           = field_data.index.day
#%% Pull NOAA COOPs data

# Set NOAA CO-OPs Station
st      = nc.Station(st_id)

# Pull water level data
st_wl   = st.get_data(
    begin_date=str_date, 
    end_date=end_date,
    product='water_level',
#    product='hourly_height',
    datum=datum, 
#    interval ='6-Minute',
    units='english',
    time_zone='gmt')

# Reset index and drop nans
st_wl   = st_wl.reset_index(drop=False)
st_wl   = st_wl.dropna()

# Check if data has duplicates and drop them out of the dateframe
st_wl   = st_wl.drop_duplicates(subset='t', keep='first')

# Only keep highs, drop out lows
# st_wl = st_wl[st_wl['v'] >= st_wl['v'].mean()]

# Remove data before and after certain dates
st_wl   = st_wl[~(st_wl['t'] < field_data.index[0])]
st_wl   = st_wl[~(st_wl['t'] > field_data.index[-1])]

# Set datetime as index
st_wl   = st_wl.set_index('t')
#%% Specify plot titles and legend items

# Plot titles, grabs start/end year from dataframe and converts to string
titles = [f"Water Level Comparison - {field_data['month'][0]}/{field_data['day'][0]}/{field_data['year'][0]}" 
              f" to {field_data['month'][-1]}/{field_data['day'][-1]}/{field_data['year'][-1]}"]

# Axes and legend items
axes_text = ['Water Level (ft, NAVD)','Water Level (m, NAVD)']
leg_text  = ['NOAA Data', 'Field Data']
#%% Plot water levels

# Create figure
f, ax = plt.subplots(1, 1, sharex=True, figsize=(15,6)) 

# Plot the datasets
# NOAA data
ax.plot(st_wl.index, st_wl['v'],'k--', linewidth=1.5, label=leg_text[0], zorder=0)

# Field data
ax.plot(field_data.index, field_data['Water Level (ft, NAVD)'],'b', linewidth=1.5, 
        label=leg_text[1], zorder=1)

# Set Titles
plt.suptitle(titles[0], fontsize=30, y=0.95)
ax.legend(ncol=len(leg_text), loc='lower center', fontsize = 12, bbox_to_anchor=(0.5,-0.03))

# Label x-axis 
# plt.xlabel('Date', fontsize=18)

# Set x-axis range
# ax.set_xlim([datetime(2024, 2, 8), datetime(2024, 2, 25)])

# Set the second y axis to show value in both feet and meters
secax = ax.secondary_yaxis('right', functions=(ft_to_m, m_to_ft))

# Label each y-axis
ax.set_ylabel(axes_text[0], fontsize=18)
secax.set_ylabel(axes_text[1], fontsize=18)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Set y-axis limits for the plot
# ax.set_ylim(-15,15)

# Rotate ticks to 45 degrees
plt.xticks(rotation=45)

# Set xticks to 7 day interval
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

# Plot dashed gridlines
ax.grid(color='black', linestyle='--')

# Save the figure
plt.savefig(fig_out[0], bbox_inches='tight')
#%% Output water level data to csv for tidal datum calcs

# Rename column headers
st_wl = st_wl.rename(columns={'v':'Water Level (ft, NAVD)'})

# Specify columns to ouput for the csv file
header   = ['Water Level (ft, NAVD)']

# Write field data to csv file
field_data.to_csv(csv_out[0], columns=header)

# Write noaa data to csv file
st_wl.to_csv(csv_out[1], columns=header)