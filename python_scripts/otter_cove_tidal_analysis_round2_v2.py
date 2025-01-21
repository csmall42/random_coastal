# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:19 2025

@author: csmall

ACAD - Sensor Analysis

Author: Chris Small, Coastal Engineer
Email: csmall@eaest.com
Organization: EA Engineering
Date Created: 1/6/2024
Date Revised: 1/21/2024
    
BELOW HAS NOT BEEN FULLY UPDATED
    
This script perfroms the follow up data analysis for the sensors deployed at 
ACAD. Handles water level (RBR) and barometric (HOBO) field data and downloads 
NOAA water level, USGS discharge, and NOAA precipitation data. Field data is 
expected to be RBR rsk files and a csv for barometric, headers MUST match. The
csv headers will most likely change, the rsk headers are defualt and should
not need to be changed.

Expects field data to be datetime in GMT/UTC, water level to be meters, and
barometrc pressure to be Hg. Plots a comparison of the data, outputs csv files,
and calculates tidal datums.

Tidal datum calculation requires: https://github.com/NOAA-CO-OPS/CO-OPS-Tidal-Analysis-Datum-Calculator

The folder structure is auto generated from the script, user just has to save
the script into the desired location and specify the path to this location. 

Folder structure should be similar to below, but can vary.

Folder structure when this code was written:

    Parent folder: ACAD
        Sub folder: scripts
            otter_cove_tidal_analysis_round2_v1.py saved in this folder
        Sub folder: analysis
            Sub folder: Dec_2024
                Sub folder: north_sensor
                    213103_20250109_1152_north_sensor.rsk
                        Header 1 - 'timestamp'
                        Header 2 - 'depth'
                Sub folder: south_sensor
                    208699_20250109_1236_south_sensor.rsk
                        Header 1 - 'timestamp'
                        Header 2 - 'depth'
                Sub folder: baro_sensor
                    acad_baro.csv
                        Header 1 - 'Date Time, GMT+00:00'
                        Header 2 - 'Pressure, in Hg (LGR S/N: 21480517, SEN S/N: 21490495, LBL: acad)'
"""
#%% Import required packages
import os
import noaa_coops as nc
from pyrsktools import RSK
import pandas as pd
from dataretrieval import nwis
from meteostat import Stations, Hourly, units
# from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as image
# from scipy.signal import savgol_filter
import glob
import subprocess
# Path to tidal datum analysis calculator scripts
tdac_path = r'\Python\CO-OPS-Tidal-Analysis-Datum-Calculator-main'
#%% User Input Section

# Specify folder where script is saved
os.chdir(r'\Python\NPS\ACAD\Otter_Creek\Scripts')

# Set working directory to be one folder up
os.chdir('../')

# Set file ID and path to files
file_id       = [os.path.join(os.getcwd(),'Analysis','Dec_2024','south_sensor',
                            'south_sensor.rsk'),
               os.path.join(os.getcwd(),'Analysis','Dec_2024','north_sensor',
                            'north_sensor.rsk'),
               os.path.join(os.getcwd(),'Analysis','Dec_2024','baro_sensor',
                            'acad_baro.csv'),
               os.path.join(os.getcwd(),'Analysis','Dec_2024','baro_sensor',
                            'not_a_file_type.dummy')] 

# Number of rows to skip at start of csv (up to the header line)
skiprows    = 1

# Output directory for the figures
fig_path    = os.path.join(os.getcwd(),'Analysis','figures')

# Check if path already exists, if not create the path
os.makedirs(fig_path, exist_ok=True)

# Output directory for csv files
csv_path    = os.path.join(os.getcwd(),'Analysis','outputs')
    
# Check if path already exists, if not create the path
os.makedirs(csv_path, exist_ok=True)

# Define headers of interest in the field files, must match exactly
# 'Depth' from RBR file is the defualt
headers     = ['depth', 'Pressure, in Hg (LGR S/N: 21480517, SEN S/N: 21490495, LBL: acad)']

# Create dictionary to define adjustment to navd88 for each sensor 
to_navd = {}

# Define adjustment values for south sensor, north sensor
# Make sure these match part of the file name
to_navd = {
    'south_sensor': -6.75, # feet, includes pvc adjustment
    'north_sensor': -4.60, # feet, includes pvc adjustment
    }

# Define location coordinates for the site
lat, lon = 44.3173, -68.1996

# Define start and end dates for field sensors (in UTC/GMT)
field_start = '20241114 21:00' # set to match field data
field_end   = '20241213 19:30' # set to match field data

# Enter NOAA CO-OPS Station ID
st_ids       = [8413320] # Bar Harbor, ME
noaa_units   = 'english' # define units: 'english' or 'metric'
datum        = 'mllw' # change as needed, NAVD is most common and preferred
mllw_to_navd = 5.945 # feet, mllw to navd88 adjustment
products     = ['water_level'] # specifify products of interest 

# Enter USGS gauge ID and parameter codes to retrieve data
usgs_sites       = ['01022840'] # Otter Creek, ME
parameter_codes  = ['00060'] # Discharge

# Output names for the figures and specify paths
fig_names   = ['acad_full_period_v1.png', 'acad_zoom_v1.png'] 
fig_out     = [os.path.join(fig_path, name) for name in fig_names]

# Output names for the csv files and specify paths
csv_names   = ['south_sensor_data.csv', 'north_sensor_data.csv',
               'noaa_data.csv'] 
csv_out     = [os.path.join(csv_path, name) for name in csv_names]

# Load graphics for inclusion on figures
# north arrow and EA logo
im = [image.imread(r'Python\Graphics\add_ons\north_arrow.png'),
                   image.imread(r'\Python\Graphics\ea_logos\ea_logo_blue_name_to_side.png')]
#%% Define unit conversion functions

# Define conversion functions btw feet and meters
def ft_to_m(x):
    return x*0.3048

def m_to_ft(x):
    return x*3.2808

# Define conversion functions btw cfs and cms
def cfs_to_cms(x):
    return x*0.028317

def cms_to_cfs(x):
    return x*35.31467

# Define conversion functions btw inches and millimeters
def in_to_mm(x):
    return x*25.4

def mm_to_in(x):
    return x*0.03937

# Define conversion functions btw in Hg and dbar
def hg_to_dbar(x):
    return x*0.338639

def dbar_to_hg(x):
    return x*2.953
# %% Define general function to do conversions

# Unit conversion constants
conversions = {
    'ft_to_m': 0.3048,
    'm_to_ft': 3.2808,
    'cfs_to_cms': 0.028317,
    'cms_to_cfs': 35.31467,
    'in_to_mm': 25.4,
    'mm_to_in': 1 / 25.4,
    'hg_to_dbar': 0.338639,
    'dbar_to_hg': 2.953,
}

# Define generic unit conversion function
def convert_units(value, conversion_type):
    """
    Convert a value based on the specified conversion type.

    Parameters:
        value (float or array-like): The value(s) to convert.
        conversion_type (str): The type of conversion (e.g., 'ft_to_m').

    Returns:
        float or array-like: Converted value(s).
    """
    # Ensure the conversion type exists in the conversions dictionary
    if conversion_type not in conversions:
        raise ValueError(f"Invalid conversion type: {conversion_type}. Choose from {list(conversions.keys())}.")
        
    # Perform the conversion by multiplying the value with the corresponding conversion factor
    return value * conversions[conversion_type]
#%% Import field data from csv and rbr files

# Initialize dictionary to store data
field_data = {}

# Loop through each file
for file in file_id:
    
    # Extract just the filename for the dictionary key and file extension for loop
    file_name       = os.path.splitext(os.path.basename(file))[0] # removes extension
    file_extension  = os.path.splitext(file)[1].lower() # grabs extension

    # Import csv files
    if file_extension == '.csv':
        try:
                      
            # Import data from csv file
            file_data                  = pd.read_csv(file, skiprows=skiprows)
            
            # Convert date/time column to datetime for plotting
            file_data['datetime_gmt']  = pd.to_datetime(file_data['Date Time, GMT+00:00'])
            
            # Drop nans
            # file_data                  = file_data.dropna(subset=['water_level_ft_navd'])
            
            # Set datetime as index
            file_data                  = file_data.set_index('datetime_gmt')
            
            # Remove data before and after certain dates
            original_len    = len(file_data)
            file_data       = file_data[(file_data.index >= field_start) & 
                                      (file_data.index <= field_end)]

            # Print removed data count
            removed_count   = original_len - len(file_data)
            print(f"Removed {removed_count} data values from {file_name}.")
            
            # Split date/time into year, month, and day
            file_data['year']          = file_data.index.year
            file_data['month']         = file_data.index.month
            file_data['day']           = file_data.index.day
                                           
        # Error handling 
        except Exception as e:
            print(f"Error processing CSV file {file_name}: {e}")
            continue

    # Handle RSK files
    elif file_extension == '.rsk':
        try:
            with RSK(file) as rsk:
                
                # Read data based on field start and end datetimes
                rsk.readdata(pd.to_datetime(field_start), pd.to_datetime(field_end))
                
                # Derive sea pressure from total pressure
                rsk.deriveseapressure()
                # rsk.deriveseapressure(patm = patm_sh)
                
                # Derive depth
                rsk.derivedepth()
                
                # Optionally print list the channels (remove if not needed)
                # rsk.printchannels()
                
                # Optionally plot the data (remove if not needed)
                # fig, axes = rsk.plotdata(channels=['depth'])
                # plt.show()
                              
                # Convert RSK data to a DataFrame
                file_data = pd.DataFrame(rsk.data)
                
                # Set datetime as index
                file_data = file_data.set_index('timestamp')
                
                # Convert depth from meters to feet
                file_data['depth_ft'] = convert_units(file_data['depth'], 'm_to_ft')
                
                # Adjust water depth to water level above NAVD88 based on sensor
                sensor_found = False  # Track if a matching sensor is found

                for sensor in to_navd.keys():
                    if sensor in file_name.lower():
                        
                        # Conversion from water depth to water level
                        file_data['water_level_ft_navd'] = file_data['depth_ft'] + to_navd[sensor]
                        sensor_found = True
                        print(f"Converted RBR {sensor} water depth to water level above NAVD.")
                        break  # Exit the loop once a match is found
                
                if not sensor_found:
                    print(f"Warning: No matching sensor found for {file_name}, skipping NAVD adjustment.")
                    continue
                
        # Error handling        
        except Exception as e:
            print(f"Error processing RSK file {file_name}: {e}")
            continue

    # Skip unsupported file types
    else:
        print(f"Skipping unsupported file: {file}")
        continue
    
    # Ceate dictionary if it doesn't exist
    if file_name not in field_data:
        field_data[file_name]        = {}
    
    # Add product specific data and calc stats  
    for header in headers:
        if header in file_data.columns:
            field_data[file_name] = {
            'data': file_data, # drop in field data 
            'start_time': file_data.index[0], # pull first datetime value
            'end_time': file_data.index[-1], # pull last datetime value
            f"max_datetime_{header[0:5]}": file_data[header].idxmax(), 
            f"min_datetime_{header[0:5]}": file_data[header].idxmin(),
            f"max_value_{header[0:5]}": file_data[header].max(),
            f"min_value_{header[0:5]}": file_data[header].min(),
            f"avg_value_{header[0:5]}": file_data[header].mean(),     
    }

# Create variable for keys
field_keys  = list(field_data.keys())
#%% Pull NOAA COOPs data

# Download NOAA station data
noaa_data = {}

# Loop through each NOAA station and product
for st_id in st_ids:
    for product in products:
        
        # Create station object
        st = nc.Station(st_id)
        
        # Product-specific parameters
        product_params = {
            'water_level': {'product': product, 'datum': datum},
            'salinity': {'product': product}
        }
        
        # Water level and salinity data pull        
        st_data = st.get_data(
            begin_date=field_start, 
            end_date=field_end,
            **product_params[product],
            units='english',
            time_zone='gmt'
        )
            
        # Add loop for conversion if datum does not equal navd88
        if datum != 'navd':
            if 'v' in st_data and mllw_to_navd is not None:
                
                # Convert water level from MLLW to NAVD
                st_data['water_level_ft_navd'] = st_data['v'] - mllw_to_navd
                print(f"Converted water level from MLLW to NAVD for NOAA station: {st_id}.")
                            
            else:
                print(f"Skipping MLLW to NAVD conversion for NOAA station: {st_id} due to no conversion value provided.")
    
        # Reset index, drop nans, and remove any duplicates
        column_name = 'v' if product == 'water_level' else 's'
        st_data     = (
                st_data.reset_index(drop=False)
                        .dropna(subset=[column_name])
                        .drop_duplicates(subset='t')
        )
        
        # Set datetime as index
        st_data = st_data.set_index('t')
        
        # Remove data before and after certain dates
        original_len    = len(st_data)
        st_data         = st_data[(st_data.index >= field_data[field_keys[0]]['data'].index[0]) & 
                                  (st_data.index <= field_data[field_keys[0]]['data'].index[-1])]
    
        # Print removed data count
        removed_count   = original_len - len(st_data)
        print(f"Removed {removed_count} data values from NOAA station: {st_id}, product: {product}.")
    
        # Store data in a dictionary
        if st_id not in noaa_data:
            # Ceate dictionary if it doesn't exist
            noaa_data[st_id]      = {}
        
        # Add product specific data and calc stats
        noaa_data[st_id][product] = {
            'data': st_data,
            'start_time': st_data.index[0], # pull first datetime value
            'end_time': st_data.index[-1], # pull last datetime value
            f"max_datetime_{column_name}": st_data[column_name].idxmax(),
            f"min_datetime_{column_name}": st_data[column_name].idxmin(),
            f"max_value_{column_name}": st_data[column_name].max(),
            f"min_value_{column_name}": st_data[column_name].min(),
            f"avg_value_{column_name}": st_data[column_name].mean(), 
        }

# Create variable for keys
noaa_keys   = list(noaa_data.keys())
#%% Pull nearby weather data using meteostat

# Get nearby weather stations
stations    = Stations().nearby(lat, lon)
station     = stations.fetch(1)  # Fetch closest station

# Fetch and process hourly data
mt_data     = Hourly(station, pd.to_datetime(field_start), pd.to_datetime(field_end))

# Convert units, normalize, and interpolate up to 6 missing consecutive records
mt_data     = (
            mt_data.convert(units.imperial)
                    .normalize()
                    .interpolate(6)
            )

# Create dataframe and drop nans with 0
mt_data     = (
            mt_data.fetch()
                    .dropna(subset=['prcp'])
            )

# Remove data before and after certain dates
original_len    = len(mt_data)
mt_data         = mt_data[(mt_data.index >= field_data[field_keys[0]]['data'].index[0]) & 
                          (mt_data.index <= field_data[field_keys[0]]['data'].index[-1])]

# Print removed data count
removed_count   = original_len - len(mt_data)
print(f"Removed {removed_count} data values from meteostat station: {station.name[0]}.")

# Setup dictionary for data
meteo_data = {}

# Add product specific data and calc stats
meteo_data[f"{station.index[0]}"] = {
    'data': mt_data,
    'start_time': mt_data.index[0], # pull first datetime value
    'end_time': mt_data.index[-1], # pull last datetime value
    f"max_datetime_{'prcp'}": mt_data['prcp'].idxmax(),
    f"min_datetime_{'prcp'}": mt_data['prcp'].idxmin(),
    f"max_value_{'prcp'}": mt_data['prcp'].max(),
    f"min_value_{'prcp'}": mt_data['prcp'].min(),
    f"avg_value_{'prcp'}": mt_data['prcp'].mean(), 
} 

# Create variable for keys
meteo_keys  = list(meteo_data.keys())

# # Plot quick line graph to view data, uncomment to show
# mt_data.plot(y=['prcp']) 
# plt.show()
#%% Pull USGS gauge data

# Setup logic to handle formatting field start and ends times into USGS format
# Convert to datetime object
field_start_dt  = pd.to_datetime(field_start, format='%Y%m%d %H:%M')
field_end_dt    = pd.to_datetime(field_end, format='%Y%m%d %H:%M')


# Reformat to the desired string format
usgs_start      = field_start_dt.strftime('%Y-%m-%d')
usgs_end        = field_end_dt.strftime('%Y-%m-%d')

# Retrieve the usgs data
us_data = nwis.get_record(
    sites=usgs_sites[0], 
    parameterCd=parameter_codes[0], 
    start=usgs_start, 
    end=usgs_end) 

# Create datetime index with timezone removed
us_data.index   = us_data.index.tz_convert(None)

# Remove data before and after certain dates
original_len      = len(us_data)
us_data         = us_data[(us_data.index >= field_data[field_keys[0]]['data'].index[0]) & 
                          (us_data.index <= field_data[field_keys[0]]['data'].index[-1])]

# Print removed data count
removed_count   = original_len - len(us_data)
print(f"Removed {removed_count} data values from USGS site: {us_data['site_no'][0]}.")

# Setup dictionary for data
usgs_data = {}

# Add product specific data and calc stats
usgs_data[f"{us_data['site_no'][0]}"] = {
    'data': us_data,
    'start_time': us_data.index[0], # pull first datetime value
    'end_time': us_data.index[-1], # pull last datetime value
    f"max_datetime_{parameter_codes[0]}": us_data[parameter_codes[0]].idxmax(),
    f"min_datetime_{parameter_codes[0]}": us_data[parameter_codes[0]].idxmin(),
    f"max_value_{parameter_codes[0]}": us_data[parameter_codes[0]].max(),
    f"min_value_{parameter_codes[0]}": us_data[parameter_codes[0]].min(),
    f"avg_value_{parameter_codes[0]}": us_data[parameter_codes[0]].mean(), 
} 

# Create variable for keys
usgs_keys  = list(usgs_data.keys())
#%% Specify plot/sublot titles and axis/legend labels

# Plot titles
titles          = ['ACAD: Otter Cove - Data Analysis']

#Subplot titles
sub_titles      = ['Precipitation', 'Flow Rate', 'Water Level']

# Axes labels
axes_text  = ['prcp (in)', 'prcp (mm)', 'flow (ft$^{3}$/s)', 'flow (m$^{3}$/s)',
              'wl (ft, navd)', 'wl (m, navd)']

# Legend labels
leg_text        = [f"{station.name[0]}, {station.region[0]}", 'Otter Creek, ME',
                   f"{st.name}, {st.state}", 'Field - North', 'Field - South']
#%% Plot precipitation, salinity, and water level
# # set date format for all following plots
# date_form = dates.DateFormatter("%H:%M \n %m/%d")

# # set default plot style
# plt.style.use('ggplot')

# Create figure
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,10)) 

# First subplot
l0, = ax[0].plot(meteo_data[meteo_keys[0]]['data'].index, 
                 meteo_data[meteo_keys[0]]['data']['prcp'], 'k', 
                 label=leg_text[0], zorder=0)

# Second subplot
l1, = ax[1].plot(usgs_data[usgs_keys[0]]['data'].index, 
                 usgs_data[usgs_keys[0]]['data']['00060'],'b', 
                 linewidth=1.5, label=leg_text[1], zorder=1)

# # Third subplot
l4, = ax[2].plot(noaa_data[noaa_keys[0]]['water_level']['data'].index, 
                  noaa_data[noaa_keys[0]]['water_level']['data']['water_level_ft_navd'],
                  'r', linewidth=1.5, label=leg_text[2], zorder=0)

l5, = ax[2].plot(field_data[field_keys[0]]['data'].index, 
                 field_data[field_keys[0]]['data']['water_level_ft_navd'],'b', 
                 linewidth=1.5, label=leg_text[3], zorder=0)

l6, = ax[2].plot(field_data[field_keys[1]]['data'].index, 
                 field_data[field_keys[1]]['data']['water_level_ft_navd'],'m', 
                 linewidth=1.5, label=leg_text[4], zorder=1)

# Set Titles
plt.suptitle(titles[0], fontsize=30, y=0.95)
# ax.legend(ncol=len(leg_text), loc='lower center', fontsize = 12, bbox_to_anchor=(0.5,-0.03)) 
    
# Set the second y axis to show metric values
secax0 = ax[0].secondary_yaxis('right', functions=(in_to_mm, mm_to_in))
secax1 = ax[1].secondary_yaxis('right', functions=(cfs_to_cms, cms_to_cfs))
secax2 = ax[2].secondary_yaxis('right', functions=(ft_to_m, m_to_ft))

# Label each y-axis
ax[0].set_ylabel(axes_text[0], fontsize=18)
secax0.set_ylabel(axes_text[1], fontsize=18)

ax[1].set_ylabel(axes_text[2], fontsize=18)
secax1.set_ylabel(axes_text[3], fontsize=18)

ax[2].set_ylabel(axes_text[4], fontsize=18)
secax2.set_ylabel(axes_text[5], fontsize=18)

# Add grid lines for each subplot
for axis in ax:
    axis.grid(color='black', linestyle='--')

# Rotate ticks to 45 degrees, set xticks to XX day interval, and set range
# plt.xticks(rotation=45, fontsize=12)
ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=15))

# Set global legend, collect unique legend lines and labels
lines, labels = [], []
for axis in ax:
    line, label = axis.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

# Remove duplicates by using a dictionary
unique_legend = dict(zip(labels, lines))

# Create global legend
fig.legend(unique_legend.values(), unique_legend.keys(), loc='lower center', 
           ncol=len(unique_legend), fontsize=12, bbox_to_anchor=(0.44, 0.03))

# Add a hidden axis for common axis labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# Add the EA logo
ax_logo = fig.add_axes([0.8, 0.025, 0.15, 0.15], anchor='SE')  # Adjust position and size as needed
ax_logo.imshow(im[1]) #, alpha=0.7)
ax_logo.axis('off')

# Add disclaimer text at the bottom of the figure
fig.text(0.5, 0.02, ('Disclaimer: Any conclusions drawn from the analysis of this ' 
         'information are not the responsibility of EA, NPS, or its partners.'), 
         ha='center', fontsize=8)

# Save the figure
plt.savefig(fig_out[0], bbox_inches='tight')
#%% Output data as csv files formatted for tidal datum calculator

# Setup formatting for csv outputs
csv_time_format     = '%m/%d/%Y %H:%M'
csv_float_format    = '%.3f'
csv_header          = ['water_level_ft_navd'] # column(s) to output

# Write field data to csv file
field_data[field_keys[0]]['data'].to_csv(csv_out[0], 
                                         columns=csv_header, 
                                         float_format=csv_float_format, 
                                         date_format=csv_time_format)

field_data[field_keys[1]]['data'].to_csv(csv_out[1], 
                                         columns=csv_header, 
                                         float_format=csv_float_format, 
                                         date_format=csv_time_format)

# Write noaa data to csv file
noaa_data[noaa_keys[0]]['water_level']['data'].to_csv(csv_out[2], 
                                                      columns=csv_header, 
                                                      float_format=csv_float_format, 
                                                      date_format=csv_time_format)
#%% Calculate tidal datums 

# Path to csv files with wildcard
csv_files = os.path.join(csv_path, '*.csv')

# Loop through setting up configuration file and calculating datums
for fname in glob.glob(csv_files): 
    
    file_name   = os.path.split(fname)[-1]  # xxtract the filename from the path
    output_file = os.path.join(csv_path, f"tidal_datums_{file_name.replace('.csv', '.txt')}") # create file for ouputs
    
    with open(os.path.join(tdac_path, 'config.cfg'), "r") as file:
        lines = file.readlines()
        
        lines[8]  = 'pick_method = PolyFit \n'
        lines[26] = f"control_station = {st_ids[0]} \n"
        lines[37] = 'units = Feet \n'
        lines[22] = f"fname = {os.path.join(csv_path, file_name)} \n"
        lines[48] = 'subordinate_lon = Not Entered \n'
        lines[53] = 'subordinate_lat = Not Entered \n'
    
    with open(os.path.join(tdac_path, 'config.cfg'), "w") as file:
        file.writelines(lines)
    
    # Set directory to where SDC.py is stored 
    os.chdir(tdac_path) 
    
    # Calculate tidal datums using SDC.py script
    proc = subprocess.run(['python', 'SDC.py'], capture_output=True, text=True)
    #proc = subprocess.run(['python', os.path.join(tdac_path, 'SDC.py')], capture_output=True, text=True, check=True)
   
   # Save output to file
    with open(output_file, 'w') as out_file:
        out_file.write(f"Tidal datum calculations for: {file_name}\n\n") # write out file name
        out_file.write(f"STDOUT:\n{proc.stdout}\n") # write out calcs
    
    # Print message about return codes -1 fails, 1 is successful, others are unexpected
    if proc.returncode == -1:
        print(f"Error calculating tidal datums for {file_name}")
        print(f"STDERR: {proc.stderr}")
        print(f"STDOUT: {proc.stdout}")
        
    elif proc.returncode == 1:
        print(f"Tidal datums calculated successfully for {file_name}, output saved to {output_file}")
        
    elif proc.returncode == 0:
        print(f"Tidal datums calculated normally but unexpected return code for {file_name}")
        
    else:
        print(f"Unexpected return code {proc.returncode} for {file_name}")
        print(f"STDERR: {proc.stderr}")
        print(f"STDOUT: {proc.stdout}")        