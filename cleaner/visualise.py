'''
Visualise the processed participant data.
'''

import pickle  
import random
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

def view_proc_data(path, pt_name=None, window_hours=24, event_msgs=False):
    """
    Show a continuous segment of the processed data 
    prior to segmentation.
    """
    
    # set constants
    Y_MAX, Y_MIN = -10_000, 10_000    # the range of vertical lines
    Y_LLIM, Y_ULIM = 50, 500          # blood glucose range of axis
        
    # load the processed data and events files
    with open(path + ".pkl", 'rb') as f:
        df = pickle.load(f)
    with open(path + "_events"+ ".pkl", 'rb') as f:
        events = pickle.load(f)

    # Process the trajectory data ----------------------------------------    

    # extract specific columns
    input_df = df[[
        "date",            # timestamp
        "bg",              # CGM blood glucose reading
        "basal",           # Basal insulin dose in 5-min interval (units)
        "carbs",           # Carbohydrate consumption in 5-min interval (grams)
        "bolus",           # Bolus insulin dose in 5-min interval (units)
        "not_missing",     # Has the sample been added to fill a discontinuity?
        "PtID",            # Participant Identification
    ]].copy()
        
    # filter data from a specific participant
    pt_ids = input_df["PtID"].unique()
    selected_pt = random.choice(pt_ids)
    if pt_name is not None: selected_pt = pt_name    
    input_df = input_df[input_df["PtID"] == selected_pt]
    events = events[events["PtID"] == selected_pt]
    
    # exclude samples which have been added to fill in discontinuity
    non_missing_df = input_df.loc[input_df["not_missing"] == 1.0].to_numpy()    
    
    # get a window of data of a specified size
    int_start = np.random.randint(non_missing_df.shape[0]-24*60//5)
    int_end = int_start + window_hours * 60//5
    selected_window = non_missing_df[int_start:int_end, :]
    
    # get start and end date
    start_date = selected_window[:, 0].min()
    end_date = selected_window[:, 0].max()
    time_cond = (events["date"] > start_date) & (events["date"] < end_date)
    selected_events = events[time_cond].to_numpy()

    # process the selected events
    # remove nan entries
    # ensure most informative tag is used
    if len(selected_events) > 0:        
        hidden_events = ["nan", "<none>"]
        selected_events = selected_events[
            ~np.isin(selected_events[:, 2], hidden_events)] 
        not_nan_mask = ~pd.isnull(selected_events[:, 3])
        selected_events[:, 2][not_nan_mask] = selected_events[:, 3][not_nan_mask] 

    # extract meals and boluses
    myFmt = mdates.DateFormatter("%H:%M")
    all_meals = selected_window[selected_window[:, 3] > 0, :]
    all_boluses = selected_window[selected_window[:, 4] > 0, :]

    # Visualise the trajectory window -----------------------------------------
    
    # visualise the key information
    print('###################################')
    print('Start Point: {}'.format(int_start))
    print('Participant: {}'.format(selected_pt))
    print('Meals Events: {}'.format(all_meals[:, 3]))
    print('Bolus Events: {}'.format(all_boluses[:, 4]))    
    print('###################################')    

    fig = plt.figure(figsize=(5, 6), facecolor=(1, 1, 1))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 

    # visualise the blood measurements
    print(f'\nBlood glucose over a {window_hours}-hour period:')
    print('-----------------------------------\n')
    ax0 = plt.subplot(gs[0])
    ax0.plot(selected_window[:, 0], selected_window[:, 1], color='black')
    plt.ylabel("Blood \n Glucose (mg/dl)")
    
    # visualise and label bolus doses
    if len(all_boluses) > 0: 
        for bolus_time, bolus_amount in zip(all_boluses[:, 0], all_boluses[:, 4]): 
            ax0.plot([bolus_time, bolus_time], [Y_MIN, Y_MAX], c="red", ls='dashed')
            ax0.text(bolus_time, 400, f'{round(bolus_amount, 2)} U', size=7, 
                rotation=90, verticalalignment='center', horizontalalignment="right")

    # visualise and label meal consumption
    if len(all_meals) > 0: 
        for meal_time, meal_amount in zip(all_meals[:, 0], all_meals[:, 3]): 
            ax0.plot([meal_time, meal_time], [Y_MIN, Y_MAX], c="green", ls='dashed')  
            ax0.text(meal_time, 300, f'{round(meal_amount, 2)} g', size=7, 
                rotation=90, verticalalignment='center', horizontalalignment="right")  

    # visualise and label event messages
    if event_msgs:
        for ev in selected_events:        
            ax0.plot([ev[0], ev[0]], [Y_MIN, Y_MAX], c="black", ls='dashed')
            if len(ev[1]) > 8: ev[1] = f'{ev[1][:8]}...' 
            ax0.text(ev[0], 200, ev[1], rotation=90, verticalalignment='center',
                    horizontalalignment="right", size=7)

    legend_elements = [
        Line2D([0], [0], color='red', lw=1, label='Bolus', linestyle='dashed'),
        Line2D([0], [0], color='green', lw=1, label='Meal', linestyle='dashed'),
        Line2D([0], [0], color='black', lw=1, label='Event', linestyle='dashed'),
    ]

    # add an axis legend
    ax0.legend(handles=legend_elements, loc='upper left')
            
    # set the axis range and labels
    plt.gca().axis(ymin=Y_LLIM, ymax=Y_ULIM)
    plt.gca().xaxis.set_major_formatter(myFmt)

    # visualise basal doses algorithm
    ax1 = plt.subplot(gs[1])
    ax1.plot(selected_window[:, 0], selected_window[:, 2])
    plt.gca().axis(ymin=0, ymax=1.0)

    # combine x-axis
    plt.ylabel("Insulin \n Dosing (units)")
    plt.setp(ax0.get_xticklabels(), visible=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(hspace=.0)
    plt.xlabel("Time-of-day")
    plt.tight_layout()
    plt.show()
    
    # plt.savefig('Example_2.png')