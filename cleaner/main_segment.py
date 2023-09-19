'''
Convert the processed dataset into continuous segments for 
use in machine learning prediction.
'''
import os
import pickle
import argparse
import pandas as pd
from functools import partial
import multiprocessing as mp

from cleaner.helper import *
from cleaner.feature_rep import * 
from cleaner.demographic import *
from config import def_segment


def blood_glucose_interpolate(dataframe):
    """
    Perform log interpolation between successive 
    blood glucose measurements
    """
    
    dataframe["bg"] = np.exp(np.log(dataframe["bg"]).interpolate())    
    dataframe = dataframe.loc[dataframe["bg"].notna()]
    dataframe = dataframe.reset_index(drop=True)
    
    return dataframe    


def augment_data(dataframe, augment_options):
    """
    Perform simple augmentations to the dataset 
    to remove erroneous samples.
    """
    
    if len(augment_options) > 0:
        # NOTE: Future versions will include methods for 
        # filtering and augmenting the dataset 
        raise NotImplementedError
    
    # label samples which have not been augmented
    dataframe["augmented"] = 0.0
    dataframe["idx"] = dataframe.index
    
    # interpolate blood glucose values between samples 
    # to fill in missing data
    dataframe = blood_glucose_interpolate(
        dataframe=dataframe)

    return dataframe


def filter_data(dataframe, filter_options):
    """
    Identify samples in the dataset which do not satisfy some 
    filtering condition.
    """

    if len(filter_options) > 0:
        # NOTE: Future versions will include methods for 
        # filtering and augmenting the dataset 
        raise NotImplementedError
    
    # add empty valid column
    dataframe["valid"] = 1.0

    return dataframe    


def add_sensor_data(dataframe, data_events=None, sensor_options=[]):
    """
    Add labels of additional event data such as 
    """

    # verify sensor selection
    valid_sensors = [
        "exercise", "high_fat", "high_protein", 
        "caffeine", "alcohol", "high_exercise", 
        "low_exercise"
    ]
    non_valid_snsrs = list(set(sensor_options) - set(valid_sensors))    
    assert len(non_valid_snsrs) == 0, f"Sensor options include: {valid_sensors}"

    # make a copy of original
    df = dataframe.copy()
    orig_len = len(df)

    # add additonal parameters
    if (data_events is not None) and (len(sensor_options) > 0):

        # extract chosen sensor parameters from the event log
        chosen_events = data_events[["date"] + sensor_options]
        chosen_events = chosen_events.groupby('date', as_index=False).sum()
        chosen_events[sensor_options] = chosen_events[sensor_options].clip(upper=1)
        
        # merge the chosen metrics with the full dataframe
        # fill empty entries with False 
        df = pd.merge(df, chosen_events, on="date", how="left")
        df[sensor_options] = df[sensor_options].fillna(0.0)
    
    # if the dataframe length has changed something has gone wrong
    assert orig_len == len(df), "Dataframe length has changed."
    return df


def check_sampling(dataframe):
    """
    Measure the time between successive samples to ensure interpolated 
    data doesn't make up samples majority and convert string timestamps 
    to time of day measures between 0 and 1.
    """

    # copy the original dataframe
    df = dataframe.copy()

    # seperate out the original (not_missing == 1.0) 
    # and added samples (not_missing == 0.0) 
    not_missing = df.loc[df["not_missing"] == 1.0]
    missing = df.loc[df["not_missing"] == 0.0]

    # measure the time difference between successive samples
    not_missing['time_to_next_bg'] = pd.to_datetime(
        not_missing["date"].astype(str)
    ).diff(-1).dt.total_seconds().div(60)    
    missing["time_to_next_bg"] = 0.0
    
    # measure the reverse time difference 
    not_missing['time_to_prev_bg'] = pd.to_datetime(
        not_missing["date"].astype(str)
    ).diff(1).dt.total_seconds().div(60)
    missing["time_to_prev_bg"] = 0.0
    
    # recombine the samples
    df = pd.concat([not_missing, missing])
    df.sort_values(by='date', inplace=True, ignore_index=False)
    df = df.reset_index(drop=True)

    # verify that all successive samples are a maximum of 5-minutes apart
    time_diffs = pd.to_datetime(
        df["date"].astype(str)
    ).diff(1).dt.total_seconds().div(60)  
    err_msg = "There are missing samples in the input data."
    assert (time_diffs > 5).sum() == 0, err_msg
        
    # convert time to float between 0 and 1 (0 = midnight and 0.5 = noon)
    # this is easier for machine learning models to process
    time_from_start = df["date"] - df["date"].dt.floor('D')
    secs_from_start = (time_from_start).dt.total_seconds()
    df["date"] = np.floor(secs_from_start / 86_400 * 24) / 24

    return df


def get_participant_traj(pt_name, dataset, data_events=None, params={}):
    """
    Break a single participant's data into chunks of a specified 
    length and then obtain the state, action, reward. 
    """

    # remove chained assignment warning
    pd.set_option('mode.chained_assignment', None)  

    # ensure the correct parameters are included
    req_params = [
        "traj_len", "time_diff_thresh", "augment_options", 
        "filter_options", "sensor_options", "reward_type", 
        "state_type"
    ]
    err_msg = f'Please ensure all the parameters are included: {req_params}'
    assert set(req_params) == (set(req_params) & set(params.keys())), err_msg
    
    # set the parameters and seed information
    set_seed(seed=params.get("seed", 0))
    traj_len = params["traj_len"]                    # how many hours of data per segment
    kernel_size = (traj_len * 60) // 5               # number of timesteps per segment
    time_diff_thresh = params["time_diff_thresh"]    # maximum minutes of no data recording in a segment
    
    # filter out individual participant data
    pt_dataset = dataset.loc[dataset["PtID"] == pt_name]  

    # Supplement the dataset -------------------------------------------

    # modify the dataset 
    # (i.e. interpolate blood glucose, relabel carbohydrates)
    pt_dataset = augment_data(
        dataframe=pt_dataset,
        augment_options=params["augment_options"],
    )

    # exclude invalid samples
    pt_dataset = filter_data(
        dataframe=pt_dataset,
        filter_options=params["filter_options"])

    # add event log data as binary tags
    # (e.g. exercise, high fat meals)
    pt_dataset = add_sensor_data(
        dataframe=pt_dataset,
        sensor_options=params["sensor_options"],
        data_events=data_events)

    # convert time to numerical float between 0 and 1
    pt_dataset = check_sampling(
        dataframe=pt_dataset)

    # reorder the columns
    columns = [
        "date",                     # 0: timestamp for the samples
        "bg",                       # 1: CGM measurement in mg/dl
        "basal",                    # 2: basal dose in units
        "carbs",                    # 3: carbohydrate consumption in g
        "bolus",                    # 4: bolus dose in units
        "PtID",                     # 5: participant ID
        "weight",                   # 6: weight in kg
        "time_to_next_bg",          # 7: time to next non-added sample in mins
        "time_to_prev_bg",          # 8: time to last non-added samples in mins
        "valid",                    # 9: does sample meet filtering criteria?   
        "augmented",                # 10: is sample original or augmentation?
        "idx"                       # 11: what is index of original sample?
                                    # 12+: Additional sensor parameters 
    ]
    columns.extend(params["sensor_options"])
    pt_dataset = pt_dataset[columns].copy()
    
    # get the mean basal (as demographic information)
    basal_channel = pt_dataset["basal"]
    mean_basal = float(basal_channel.mean())   

    # Create trajectories of specified length ------------------------------

    # convert to numpy and create trajectories
    # [num_samples, kernel_size, num_columns]
    trajs = pt_dataset.to_numpy().astype(np.float32)
    trajs = np.lib.stride_tricks.sliding_window_view(
        trajs, window_shape=kernel_size+1, axis=0)
    trajs = trajs.transpose(0, 2, 1)[:, ::-1]

    # exclude trajectories where a continuous period of more than 
    # the threshold is made up of interpolated samples  
    bg_forward = np.min(trajs[:, :, 7], axis=1) >= -time_diff_thresh
    bg_backward = np.max(trajs[:, :, 8], axis=1) <= time_diff_thresh
    bg_in_range = np.logical_and(bg_forward, bg_backward)

    # also exclude samples identified to be invalid during filtering
    valid = np.sum(trajs[:, :, 9], axis=1) == (kernel_size + 1)
    valid_sample = np.logical_and(bg_in_range, valid)
    trajs = trajs[valid_sample]

    # remove non-essential metrics
    # (bg, carbs, insulin, (+ other metrics), pt_mean_basal, weight, time)
    added_metrics = list(range(12, 12 + len(params["sensor_options"])))
    full_trajs = trajs[:, :, [1, 3, 4] + added_metrics + [6, 6, 0]]
    full_trajs[:, :, -3] = mean_basal

    # extract basal and update bolus with basal to make 
    # column with full insulin information
    basal = trajs[:, :, [2]].copy()
    full_trajs[:, :, [2]] += basal

    # get unprocessed observations, actions, rewards and terminals  
    raw_obs = full_trajs.copy()
    raw_action = basal[:, :-1].copy()
    raw_reward = process_reward(
        full_trajs, reward_type=params["reward_type"])
    raw_terminal = np.zeros(raw_action.shape)

    # processing step is memory intensive and therefore requires chunking
    obs = []
    chunk_size = 5_000
    num_chunks = int(raw_obs.shape[0]/chunk_size + 1) 
    for c in range(num_chunks):

        # extract samples
        obs_inp = raw_obs[chunk_size*c:chunk_size*(c+1)]
        obs_inp = np.lib.stride_tricks.sliding_window_view(
            obs_inp, window_shape=kernel_size//2, axis=1)
        obs_inp = obs_inp.transpose(0, 1, 3, 2).reshape(
            -1, kernel_size//2, raw_obs.shape[-1])

        # condense the representation as specified 
        obs_inp, stats = condense_state(
            obs_inp, state_type=params["state_type"])
        obs_inp = obs_inp.reshape(-1, kernel_size//2+2, obs_inp.shape[-1])
        obs.append(obs_inp[:, :-1, :])
    obs = np.concatenate(obs, axis=0)

    # get processed action, reward, terminal
    action = raw_action.copy()[:, :kernel_size//2]
    reward = raw_reward[:, :kernel_size//2]
    terminal = raw_terminal[:, :kernel_size//2]

    # Save the outputs ----------------------------------------------
    
    # save the state representation
    def_start = ["bg"]
    raw_middle = ["carbs", "insulin"]
    def_end = params["sensor_options"] + ["mean_basal", "weight", "time"]

    # add information necessary for distinguishing augmentations
    # augmented = has sample been augmented?
    # idxs = what is the original index of the sample pre-augmentation?
    sample_idxs = trajs[:, 0, 11]
    augmented_samples = np.sum(trajs[:, :, 10], axis=1) > 0
    information = {
        "augmented": augmented_samples.astype(np.float32), 
        "idxs": sample_idxs.astype(np.float32)             
    }
    
    # log the processed data
    # features have been converted into condensed form
    # states contains representation information
    information["states"] = def_start+stats[-1]+def_end
    data_dict = dict(
        trajectories={
            "observations": obs.astype(np.float32),
            "actions": action.astype(np.float32),
            "rewards": reward.astype(np.float32),
            "terminals": terminal.astype(np.float32)
        },
        information=information,        
        PtID=pt_name
    )

    # log the raw data
    # features have not been converted into condensed form
    # states contains representation information
    information["states"] = def_start+raw_middle+def_end
    raw_data_dict = dict(
        trajectories={
            "observations": raw_obs.astype(np.float32),
            "actions": raw_action.astype(np.float32),
            "rewards": raw_reward.astype(np.float32),
            "terminals": raw_terminal.astype(np.float32)
        },
        information=information,        
        PtID=pt_name
    )

    return data_dict, raw_data_dict


def init_globals_clean(counter):
    """
    Define the global variables for threaded data cleaning.
    """
    global _counter_collect
    _counter_collect = counter 


def multi_get_pt_traj(pt_name, dataset, data_events=None, params={}):
    """
    Use multiprocessing to break a participants diabetes
    data into chunks of specified length.
    """

    # get the processed and unprocessed segments
    # from the individual participant
    data_dict, raw_data_dict = get_participant_traj(
        pt_name=pt_name, 
        dataset=dataset,   
        data_events=data_events,     
        params=params
    )

    # update the global counter
    num_pts = len(list(dataset["PtID"].unique()))
    with _counter_collect.get_lock():
        _counter_collect.value += 1
        num_samples = len(data_dict["trajectories"]["observations"])
        print(f'{_counter_collect.value}/{num_pts} - Completed: {pt_name} with {num_samples} samples')
        
    return data_dict, raw_data_dict

def break_trajectories(
    dataset_path, demographic_path, pt_file_path=None, 
    save_path=None, num_jobs=8,  params={}):  
    """
    Break a dataset collected from multiple participants into 
    segments suitable for use as the input to a machine learning model.
    """
    
    # load the processed dataset
    with open(dataset_path + ".pkl", 'rb') as f:
        dataset = pickle.load(f)
    dataset.sort_values(by='date', inplace=True, ignore_index=False)
    dataset = dataset.reset_index(drop=True)
    
    # load the participant ids
    demo_file = load_demographic_data(
        demographic_path, pt_file=pt_file_path)
    pt_names = list(demo_file["PtID"].unique())
    
    # filter the dataset (if selecting a subset)
    dataset = dataset[dataset['PtID'].isin(pt_names)].copy()
    
    # load the events dataframe
    data_events = None
    if os.path.exists(f'{dataset_path}_events.pkl'):
        with open(f'{dataset_path}_events.pkl', 'rb') as f:
            data_events = pickle.load(f)
        data_events = data_events[data_events['PtID'].isin(pt_names)].copy()

    # Perform multiprocessing loop -----------------------------------
    
    # instantiate the inputs for the multiprocessing loop
    multi_func = partial(
        multi_get_pt_traj, 
        dataset=dataset,
        data_events=data_events,
        params=params
    )
    
    # initialise counter and pooling queue
    counter_obj = mp.Value('i', 0)
    pool = mp.Pool(
        min(num_jobs, len(pt_names)),
        initializer=init_globals_clean, 
        initargs=(counter_obj,)
    )
    
    # start with largest participant dataset
    pt_names = dict(dataset['PtID'].value_counts())
    pt_names = list(pt_names.keys())

    # segment the patient data
    pt_files, raw_pt_files = zip(*pool.map(multi_func, pt_names))
 
    # save a processed version
    if save_path != None:
        full_save_path = f'{save_path}.pkl'
        with open(full_save_path, 'wb') as f:
            pickle.dump(pt_files, f)

        # save an unprocessed version
        raw_save_path = f'{save_path}_raw.pkl'
        with open(raw_save_path, 'wb') as f:
            pickle.dump(raw_pt_files, f)
    
    return pt_files, raw_pt_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="location of the raw data file folder")
    parser.add_argument("--save_path", type=str, help="location and name of save file")
    parser.add_argument("--demo_path", type=str, help="location of the demographic file")
    parser.add_argument("--params", type=str, help="parameters for segmentation")
    parser.add_argument("--num_jobs", type=int, default=8, help="number of threads for multiprocessing")
    args = parser.parse_args()

    # load in  the default parameters
    config = parse_config(def_segment)
    args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(args)
    
    # process raw data files
    break_trajectories(
        dataset_path=config["dataset_path"],
        demographic_path=config["demo_path"],
        save_path=config["save_path"],
        pt_file_path=config["pt_path"],
        num_jobs=config["num_jobs"],
        params=config["params"]

    )