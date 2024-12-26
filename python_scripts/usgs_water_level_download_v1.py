# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:37:37 2024

@author: csmall
"""

# Script to download USGS water level data

#%% Import required packages
from dataretrieval import nwis

#%% User input section

# Enter USGS gauge ID and other parameters to retrieve data
site            = '01305575' # Great South Bay at Watch Hill, FIIS, NY
parameterCode   = '62620' # water level
startDate       = '2024-02-09'
endDate         = '2024-05-01'

#%% Pull USGS gauge data

# Retrieve the data, units should be feet but double check
usgs_wl = nwis.get_record(sites=site, parameterCd=parameterCode, start=startDate, end=endDate) 
print('Retrieved ' + str(len(usgs_wl)) + ' data values.')

# Create new column with converted utc time (removing timestamp)
# May not be needed, was kicking an error message at one point
usgs_wl['datetime_utc'] = usgs_wl.index.tz_convert(None)

# Reset dataframe index to newly created datetime
usgs_wl           = usgs_wl.set_index('datetime_utc')

# Quick plot of data
ax = usgs_wl.plot(y=parameterCode+'_navd88')
ax.set_xlabel('Date')
ax.set_ylabel('Water Level (ft, NAVD)')
