'''
Convert the quantitative type 1 diabetes metrics, such as blood glucose,
insulin doses and carbohydrate consumption into a complete dataset.
'''

import os
import re 
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import multiprocessing as mp
from functools import partial

from cleaner.demographic import *


def extract_user_files(full_path):
    """
    Convert a folder of patient files into seperate dataframes for 
    each core file types: devicestatus, entries, profile, treatments
    """
    
    # list all files in folder and trim to distinct type
    full_filenames = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]  
    filenames = [re.split('_|\.', f)[0] for f in full_filenames]
            
    # get file name and number if there are multiple files
    file_keys = list(Counter(filenames).keys()) 
    file_count = list(Counter(filenames).values()) 

    # group files by type
    all_files = {"devicestatus":[], "entries":[], "profile":[], "treatments":[]}
    file_count = [c for _, c in sorted(zip(file_keys, file_count))]
    full_filenames.sort()
    file_keys.sort()  
        
    # create dataframe and assign file to group based on type
    counter = 0
    for idx, k in enumerate(file_keys):
        for _ in range(file_count[idx]):   
            file_name = full_filenames[counter] 
            dataframe = pd.read_json(f'{full_path}/{file_name}', convert_dates=False)             
            all_files[str(k)] += [dataframe]
            counter += 1
        
    # combine dataframes of the same type
    entries = pd.concat(all_files["entries"], ignore_index=True, sort=False)
    treatments = pd.concat(all_files["treatments"], ignore_index=True, sort=False)
    profile = pd.concat(all_files["profile"], ignore_index=True, sort=False)
    devicestatus = pd.concat(all_files["devicestatus"], ignore_index=True, sort=False)
    
    return entries, treatments, profile, devicestatus


"""
Convert the raw OpenAPS json files of a single participant 
to a complete dataframe.
"""
def json_to_dataframe(pt_name, dataset_path, debug=0):
        
    # remove cap on number of columns and hide warning message
    pd.set_option('display.max_rows', 10_000, 'display.max_columns', None)
    pd.set_option('mode.chained_assignment', None)   
    
    # get all files in selected folder
    full_path = dataset_path + pt_name + '/direct-sharing-31/'
    entries, treatments, profile, devicestatus = extract_user_files(full_path)
        
    # process the entries -----------------------------------------------------
          
    bg = convert_entries(entries)
            
    if debug > 2: 
        print('\nEntries columns: -------------------')
        print(list(entries.columns))
        
    # process the profile ------------------------------------------------------
    
    prof = convert_profile(profile, treatments)
    
    if debug > 2: 
        print('\nProfile columns: -------------------')
        print(list(profile.columns)) 
        print(prof["basal"].values)
    
    # process the treatments ---------------------------------------------------
                
    basal, bolus = convert_treatments(treatments, prof)
    
    if debug > 2:
        print('\nTreatments columns: -------------------')
        print(list(treatments.columns))    
        print(treatments["eventType"].value_counts())
        if 'notes' in treatments.columns:
            print(treatments["notes"].value_counts())
        print('----------------------------------------')
            
    if debug > 1:
        print('\nSummary of Patient ------------------')
        print(" - {} days of data".format(round((len(bg) * 5) / (60 * 24), 2)))
        print(" - {} meals occured".format(len(bolus[bolus["carbs"] > 0])))
        print(" - Mean {} meals per day".format(round(len(bolus[bolus["carbs"] > 0])/((len(bg) * 5)/(60 * 24)), 2)))
        print('--------------------------------------\n')
    
    # merge the data ---------------------------------------------------------
        
    final_dataframe = merge_data_streams(
        bg, basal, bolus, prof, debug=debug
    )
    
    if debug > 1:
        # display action distribution
        print(f'\nmean action {np.mean(final_dataframe["basal"].values)}')
        print(f'median action {np.median(final_dataframe["basal"].values)}\n')
        plt.hist(final_dataframe["basal"].values, bins=100)
        plt.show()
        
        # display the nans in each column
        print('\nNaNs per column: -----------------')
        print(final_dataframe.isnull().sum()) 
        print('--------------------image.png----------------')
        
    # save all the event alerts and their times
    event_msgs = process_event_msgs(treatments)
            
    return final_dataframe, event_msgs



def multi_js_to_df(pt_id, dataset_path, demographic_file):
    """
    Convert the raw individual patient device log JSON 
    file into a dataframe.
    """

    #################################
    # TODO: want to remove debugging
    #################################

    # convert JSON file to dataframe
    final_df, event_msgs = json_to_dataframe(
        pt_id, dataset_path=dataset_path, debug=1) 

    # add demographic information


    # add additional columns
    final_df["PtID"] = pt_data[0]
    final_df["weight"] = pt_data[1] 
    final_df["height"] = pt_data[2]
    final_df["age"] = pt_data[3]
    final_df["gender"] = pt_data[4]
    final_df["daily_carbs"] = pt_data[5]


    event_msgs["PtID"] = pt_data[0]
        
    # update the run counter
    with _counter_collect.get_lock():
        _counter_collect.value += 1
        num_pts = len(demographic_file)
        print(f'{_counter_collect.value}/{num_pts} - Completed: {pt_id}')
    
    return (final_df, event_msgs)


def init_globals_clean(counter):
    """
    Define the global variables for threaded data cleaning.
    """
    global _counter_collect
    _counter_collect = counter 


def collect_cohort(
    dataset_path, demographic_path, save_path=None, pt_file_path=None, num_jobs=8):
    """
    Extract patient device logs from raw OpenAPS device data and
    combine into a single dataframe. 
    """

    # load the demographic data
    demo_file = load_demographic_data(
        demographic_path, pt_file=pt_file_path)
    
    # extract the participant IDs and select relevant colums
    pt_ids = list(demo_file["PtID"].unique())
    demo_file = demo_file[[
        "PtID", "weight", "height", 
        "age", "gender", "daily_carbs" 
    ]]    
    
    # intialise the pooling queue
    # add a counter of completed runs
    counter_obj = mp.Value('i', 0)
    pool = mp.Pool(
        num_jobs,
        initializer=init_globals_clean, 
        initargs=(counter_obj,))
    
    # set function arguments and start run
    multi_func = partial(
        multi_js_to_df, 
        dataset_path=dataset_path,
        demographic_file=demo_file)
    pool_output = pool.map(multi_func, pt_ids)
    pool.close()
    
    # combine the individual patient dataframes
    dataset_files, event_files = zip(*pool_output)
    full_dataset = pd.concat(dataset_files)
    full_events = pd.concat(event_files)
    
    # save the dataframe
    if save_path is not None:
        with open(save_path + ".pkl", 'wb') as f:
            pickle.dump(full_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_path +"_events" + ".pkl", 'wb') as f:
            pickle.dump(full_events, f, protocol=pickle.HIGHEST_PROTOCOL)
     
    return full_dataset, full_events


if __name__ == "__main__":
    
    DEMO_PATH = './datasets/OpenAPS_Demographic.xlsx'
    DATA_PATH = './datasets/Raw_data'
    collect_cohort(
        dataset_path=DATA_PATH,
        demographic_path=DEMO_PATH
    )