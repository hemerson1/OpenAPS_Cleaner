"""
Construct the features for the input to the machine 
learning model.
"""

import numpy as np

def check_times(states, hour_steps=12):
    """
    Ensure state is continuous and spaced at 
    5-minute intervals before processing.
    """
    sample_times = states[:, :, -1].copy() 
    past_time = (sample_times[:, 0] - 60/1440) % 1     
    times_close = np.isclose(sample_times[:, hour_steps], past_time, atol=1e-3) 
    return np.all(times_close)


def magni_reward(blood_glucose):
    """
    Use the Magni risk function to calculate the reward for 
    the current blood glucose value.
    """   

    p1, p2, p3 = 3.5506, 0.8353, 3.7932
    reward = -10 * (p1 * (np.log(np.maximum(
        np.ones(1), blood_glucose[0]))**p2 - p3))**2 
    return reward


def process_reward(state, reward_type="magni"):
    """
    Calculate the reward for the control agent based
    on the state.
    """

    # ensure state processing exists within package
    valid_types = ["magni"]
    err_msg = f"Valid reward methods include: {valid_types}"
    assert reward_type in valid_types, err_msg

    if reward_type == "magni":
        kernel_size = state.shape[1]
        reward = magni_reward(state[:, 1:, [0]].reshape(1, -1, 1))
        reward = reward.reshape(-1, kernel_size-1, 1)

    return reward

def default_state(state, params={}):
    """
    Use the state implementation from:
    https://arxiv.org/abs/2204.03376

    # uses : (30-min bg over 4hrs, mob, iob, (+added metrics), mean_basal, weight, time)
    """

    # set parameters
    horizon = params["horizon"]
    hour_steps = params["hour_steps"]
    state_size = params["state_size"]
    def_state_size = params["def_state_size"]
    
    # get blood glucose at 30-minute intervals 
    bg_idxs = list(range(0, horizon, hour_steps//2)) + [horizon - 1]
    bg_intervals = state[:, bg_idxs, 0].reshape(-1, len(bg_idxs))  
    
    # get insulin-on-board with activity peak after 3 hrs 
    # this is an approximation of insulin activity
    peak_time = 3 # time to peak in hrs
    max_dur = peak_time * hour_steps
    peak_dur = peak_time * hour_steps * (75 / 180) 
    peak_increase = np.arange(0, 1, 1/peak_dur)
    peak_decrease = np.arange(1, 0, -1/(max_dur-peak_dur))
    iob_mult = np.concatenate([peak_increase, peak_decrease]) 
    iob_mult = np.pad(iob_mult, (0, horizon-len(iob_mult)), "constant")
    iob = np.sum(state[:, :, 2] * iob_mult, axis=1).reshape(-1, 1)

    # and meals-on-board (peak after 20 minutes)
    # this is an approximation of carboydrate activity
    peak_time = 4 # number of timesteps
    peak_increase = np.arange(0, 1, 1/peak_time)
    peak_decrease = np.arange(1, 0, -1/(horizon - peak_time))
    mob_mult = np.concatenate([peak_increase, peak_decrease])           
    mob = np.sum(state[:, :, 1] * mob_mult, axis=1).reshape(-1, 1)  
    
    # extract the time, weight and mean basal
    mean_basal = state[:, 0, -3].reshape(-1, 1)
    weight = state[:, 0, -2].reshape(-1, 1)
    time = state[:, 0, -1].reshape(-1, 1)

    # account for any additional sensor data included in the state
    metrics = []
    for i in range(state_size - def_state_size):
        metric_feat = state[:, 0, 3+i].reshape(-1, 1)
        metrics.append(metric_feat)

    # combine to create the state
    state_feats = [bg_intervals, mob, iob] + metrics + [mean_basal, weight, time]
    cond_state = np.concatenate(state_feats, axis=1)  

    # set optional state representations
    state_rep = ["bg (t-30)", "bg (t-60)", "bg (t-90)", "bg (t-120)",
                 "bg (t-150)", "bg (t-180)", "bg (t-210)", "bg (t-240)",  "cob", "iob"]

    return cond_state, state_rep


def zhu_state(state, params={}):
    """
    Use the state implementation from:
    https://ieeexplore.ieee.org/abstract/document/9813400

    # uses : (bg, meal, ins, mob, iob, (+added metrics), mean_basal, weight, time)
    """
    
    # set parameters
    horizon = params["horizon"]
    state_size = params["state_size"]
    def_state_size = params["def_state_size"]

    # get insulin activity
    ins_decay = np.arange(1, 0, -1/horizon)
    iob = np.sum(state[:, :, 2] * ins_decay, axis=1)
    iob = iob.reshape(-1, 1)

    # get carbohydrate activity
    carb_abs = 0.5 # g/min 
    peak_steps = 3 # or 15 mins
    time_interval = 5 # mins
    carb_decay = np.arange(0, horizon - peak_steps, 1)
    carb_decay = np.concatenate([np.zeros(peak_steps), carb_decay], axis=0)
    cob = np.sum(np.maximum(state[:, :, 1] - carb_abs * carb_decay * time_interval, 0), axis=1)
    cob = cob.reshape(-1, 1)

    # add in standard parameters
    bg = state[:, 0, 0].reshape(-1, 1)
    insulin = state[:, 0, 2].reshape(-1, 1)
    carbs = state[:, 0, 1].reshape(-1, 1)
    mean_basal = state[:, 0, -3].reshape(-1, 1)
    weight = state[:, 0, -2].reshape(-1, 1)
    time = state[:, 0, -1].reshape(-1, 1)

    # process additional metrics
    metrics = []
    for i in range(state_size - def_state_size):
        metric_feat = state[:, 0, 3+i].reshape(-1, 1)
        metrics.append(metric_feat)

    # combine to create the state
    state_feats = [bg, carbs, insulin, cob, iob] + metrics + [mean_basal, weight, time]
    cond_state = np.concatenate(state_feats, axis=1)

    # set optional state representations
    state_rep = ["carbs", "insulin", "cob", "iob"]

    return cond_state, state_rep 

def condense_state(state, state_type="default"):
    """
    Transform the (horizon, 4) state into a condensed metric incorporating
    all the important information.
    """

    # check the state dimensionality is correct
    err_msg = "State should be of shape (num_samples, kernel_size, state_dims)"
    assert state.ndim == 3, err_msg

    # ensure state processing exists within package
    valid_types = ["default", "zhu", None]
    err_msg = f"Valid condensing methods include: {valid_types}"
    assert state_type in valid_types, err_msg

    # extract the parameters needed for processing
    params = {
        "def_state_size": 6,             # (bg, carbs, insulin, pt_mean_basal, weight, time)
        "time_interval": 5,              # minutes between each sample
        "horizon": state.shape[1],       # number of samples considered in each state
        "state_size": state.shape[2],    # number of features in selected state representation
        "hour_steps":  60//5 
    }

    # correct small deviations from time via normalisation
    state[:, :, -1] = np.round(state[:, :, -1] * 24) / 24
    state[state[:, :, -1] == 1, -1] = 0 

    # check samples are continuous
    wrong_input_msg = "State input should go from latest to earliest time." 
    assert check_times(state, hour_steps=params["hour_steps"]), wrong_input_msg  
    
    # make a copy of the state
    cond_state = state.copy()

    # --------------------------------------------------------
    # select the appropriate state representation
    # rules:
    # - first feature must remain as blood glucose
    # - final three features must be mean basal, weight, time
    # --------------------------------------------------------

    state_rep = ["carbs", "insulin"]
    if state_type == "default":   
        cond_state, state_rep = default_state(
            state=state, params=params)
    elif state_type == "zhu":
        cond_state, state_rep = zhu_state(
            state=state, params=params)

                
    # get the mean and standard deviation
    # and the final state representation
    stats = [
        np.mean(cond_state, axis=0), 
        np.std(cond_state, axis=0),
        state_rep
    ]
    
    return cond_state, stats  