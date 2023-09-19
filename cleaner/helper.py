"""
Re-usable functions for dataset pre-processing.
"""

import random
import numpy as np
from collections import OrderedDict


def set_seed(seed=0):
    """
    Seed the process to create reproducible results.
    """
    np.random.seed(seed)
    random.seed(seed)


def read_selected_pts(pt_file=None):
    """
    Select participants from the OpenAPS repository.
    Due to non-standardised nature of the dataset not 
    all participants will have data in the correct format
    or with compatible devices.   
    """
    
    # use a default file
    if pt_file is None: pt_file = './datasets/pts.txt'

    # read file and convert to list
    with open(pt_file, "r") as file:
        pts = file.read().splitlines()
    return pts


def modify_date(row):       
    """
    Convert a datetime pandas element from an arbitrary 
    format to a standardised format.

    NOTE: this can be performed via pandas, but this 
    method is considerably faster

    Standardised format: yyyy-mm-ddTHH:MM:SS
    """
    
    new_row = row
            
    # e.g. 2015-08-18T00:07:15+00:00
    if len(row) == 25: 
        new_row = row[:-6]
    
    # e.g. 2015-12-31T00:26:49.676Z
    elif len(row) == 24:
        new_row = row[:-5] 
    
    # e.g. 2017-08-04T19:56:49.653-0400
    elif len(row) == 28: 
        new_row = row[:-9] 
        
    # e.g. 2015-06-07T22:52:50.173-04:00
    elif len(row) == 29:
        new_row = row[:-10] 
    
    # e.g. 2017-05-17T23:31:13Z
    elif len(row) == 20:
        new_row = row[:-1] 
    
    # e.g 2018-02-04T09:52:36.028000+01:00
    elif len(row) == 32:
        new_row = row[:-13] 
        
    # e.g.2017-07-30T23:05-0400
    elif len(row) == 21: 
        new_row = row[:-5] + ":00"
    
    # e.g. 07/26/2015 20:33:58 PM
    elif len(row) == 22:
        row_spt = row.split(" ")
        date_spt = row_spt[0].split("/")
        new_row = date_spt[2] + "-" + date_spt[0] + "-" + date_spt[1] + "T" + row_spt[1]
        
    # e.g. 2017-04-06T22:02:48.897000-0500
    elif len(row) == 31:
        new_row = row[:-12]
        
    # e.g. Thu Aug 23 21:25:44 GMT+02:00 2018
    elif len(row) == 34:
        row_spt = row.split(" ")
        year, month, day = row_spt[-1], row_spt[1], row_spt[2]
        time = row_spt[3]
        
        month_data = {
            "Jan": "01", "Feb": "02", "Mar": "03",
            "Apr": "04", "May": "05", "Jun": "06",
            "Jul": "07", "Aug": "08", "Sep": "09",
            "Oct": "10", "Nov": "11", "Dec": "12",
        }
        new_row = year + "-" + month_data[month] + "-" + day + "T" + time 
        
    return new_row

def parse_config(cfg_module):
    '''Convert configuratio file to dictionary.'''

    args = [ i for i in dir(cfg_module) if not i.startswith("__")]
    config = OrderedDict()
    for arg in args:
        config[arg] = getattr(cfg_module, arg)
    
    return config


