'''
Convert the quantitative type 1 diabetes metrics, such as blood glucose,
insulin doses and carbohydrate consumption into a complete dataset.
'''

import os
import re 
import pandas as pd
import numpy as np
from collections import Counter
import multiprocessing as mp


'''
TODO:
- how to explain function inputs and outputs?

'''

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
Retrieve multiple patients and combine them into 
a single dataset.
"""
def collect_cohort(
    patient_data, dataset_path, save_path=None, chosen_pts=[], num_jobs=8):
    
    ############################################
    # TODO: update the inputs of this function
    ############################################
    
    # get names and weights of pts
    pt_names = list(patient_data["PtID"].unique())
    pt_weights = list(patient_data["weight"])
    pt_heights = list(patient_data["height"])
    pt_ages = list(list(patient_data["age"]))
    pt_genders = list(list(patient_data["gender"]))
    pt_carbs = list(list(patient_data["daily_carbs"]))
    
    # filter participants
    if len(chosen_pts) > 0:
        pt_idxs = [i for i in range(len(pt_names)) if pt_names[i] in chosen_pts]
        pt_names = chosen_pts
        pt_weights = np.array(pt_weights)[pt_idxs]
        pt_heights = np.array(pt_heights)[pt_idxs]
        pt_ages = np.array(pt_ages)[pt_idxs]
        pt_genders = np.array(pt_genders)[pt_idxs]
        pt_carbs = np.array(pt_carbs)[pt_idxs]
        
    # Perform multiprocessing loop ---------------------------------
    
    # fill in the constant arguements
    multi_func = partial(
        multi_js_to_df, 
        dataset_path=dataset_path,
        num_pts=len(pt_names)
    )
    
    # initialise counter and pooling queue
    counter_obj = mp.Value('i', 0)
    pool = mp.Pool(
        num_jobs,
        initializer=init_globals_clean, 
        initargs=(counter_obj,)
    )
    
    # clean the patient data
    pool_output = pool.map(multi_func, list(
        zip(pt_names, pt_weights, pt_heights, pt_ages, pt_genders, pt_carbs)
    ))
    pool.close()
    
    # combine the dataframes
    dataset_files, event_files = zip(*pool_output)
    full_dataset = pd.concat(dataset_files)
    full_events = pd.concat(event_files)
    
    # save the dataframe
    if save_path != None:
        with open(save_path + ".pkl", 'wb') as f:
            pickle.dump(full_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(save_path + "_events" + ".pkl", 'wb') as f:
            pickle.dump(full_events, f, protocol=pickle.HIGHEST_PROTOCOL)
     
    return full_dataset, full_events


if __name__ == "__main__":
    pass