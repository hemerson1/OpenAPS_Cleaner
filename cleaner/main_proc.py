'''
Convert the quantitative type 1 diabetes metrics, such as blood glucose,
insulin doses and carbohydrate consumption into a complete dataset.
'''

import os
import re 
import pickle
import json
import argparse
import pandas as pd
from collections import Counter
import multiprocessing as mp
from functools import partial

from cleaner.demographic import *
from cleaner.quant_proc import *
from cleaner.qual_proc import *
from config import def_proc

# overcome loading pd error of certain type
# Value error too small when loading Json
pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

# stop memory error relating to multiprocessing
mp.set_start_method('spawn', force=True)


def extract_user_files(full_path):
    """
    Convert a folder of patient files into seperate dataframes for 
    each core file types: devicestatus, entries, profile, treatments
    """
    
    # list all files in folder and trim to distinct name format
    full_filenames = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
    full_filenames = [f for f in full_filenames if f.endswith(".json")]
    filenames = [re.split('_|\.', f)[0] for f in full_filenames]
            
    # get file name and number if there are multiple files
    file_keys = list(Counter(filenames).keys()) 
    file_count = list(Counter(filenames).values()) 

    # group files by one of four categories
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


def summarise_patient(bg, bolus, pt_name):
    """Briefly summarise key patient information"""
    
    num_days = (len(bg) * 5) / (60 * 24)
    meal_num = len(bolus[bolus["carbs"] > 0])

    print('\nSummary of Patient --------')
    print("Patient ID: {}".format(pt_name))
    print(" - {} days of data".format(round(num_days, 2)))
    print(" - {} meals occured".format(meal_num))
    print(" - Mean {} meals per day".format(round(meal_num/num_days), 2))
    print('----------------------------\n')


def json_to_dataframe(pt_name, dataset_path):
    """
    Open the indiviudal device logs:
    - entries -> Glucose Monitor
    - treatments -> Insulin Pump
    - profile -> NightScout Information

    Convert into standardised form and collate 
    into a single dataframe.
    """

    # hide warning relating to chained assignment
    pd.set_option('mode.chained_assignment', None)   
    
    # extract individual logs from user files
    full_path = dataset_path + pt_name + '/direct-sharing-31/'
    entries, treatments, profile, _ = extract_user_files(full_path)

    # process log data 
    bg = convert_entries(entries)                         # blood glucose 
    prof = convert_profile(profile, treatments)           # background basal information
    basal, bolus = convert_treatments(treatments, prof)   # basal, bolus and carbohydrates

    # merge into single dataframe with samples at 5-min intervals     
    final_dataframe = merge_data_streams(
        bg, basal, bolus, prof)

    # convert self-reported notes into binary tags
    # e.g. high fat, high protein, exercise, etc.
    event_msgs = process_event_msgs(treatments)

    # display the patient info
    summarise_patient(bg, bolus, pt_name)
            
    return final_dataframe, event_msgs


def init_globals_clean(counter):
    """
    Define the global variables for threaded data cleaning.
    """
    global _counter_collect
    _counter_collect = counter 


def multi_js_to_df(pt_id, dataset_path, demographic_file):
    """
    Convert the raw individual patient device log JSON 
    file into a dataframe.
    """

    # convert JSON file to dataframe 
    # final_df -> diabetes metrics
    # events_msgs -> qualitative event labels
    final_df, event_msgs = json_to_dataframe(
        pt_id, dataset_path=dataset_path) 

    # add demographic information to dataframe
    pt_demo = demographic_file.loc[demographic_file["PtID"] == pt_id]
    for col in list(pt_demo.columns):
        final_df[col] = pt_demo.iloc[0][col]
    event_msgs["PtID"] = pt_id
        
    # update the run counter
    num_pts = len(demographic_file)
    with _counter_collect.get_lock():
        _counter_collect.value += 1
        print(f'{_counter_collect.value}/{num_pts} - Completed: {pt_id}')
    
    return (final_df, event_msgs)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="location of the raw data file folder")
    parser.add_argument("--save_path", type=str, help="location and name of save file")
    parser.add_argument("--demo_path", type=str, help="location of the demographic file")
    parser.add_argument("--pt_path", type=int, default=None, help="location and name of participant file")
    parser.add_argument("--num_jobs", type=int, default=8, help="number of threads for multiprocessing")
    args = parser.parse_args()

    # load in  the default parameters
    config = parse_config(def_proc)
    args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(args)
    
    # process raw data files
    collect_cohort(
        dataset_path=config["dataset_path"],
        save_path=config["save_path"],
        demographic_path=config["demo_path"],
        pt_file_path=config["pt_path"],
        num_jobs=config["num_jobs"]
    )