"""
Functions for converting the raw diabetes data 
in the device logs into a complete and standardised form.
"""

import ast
import time
import datetime
import numpy as np
import pandas as pd
from functools import reduce

from cleaner.helper import *


def convert_entries(entries):
    """
    Clean the continuous glucose monitor device data 
    and extract blood glucose measurements.   
    """

    # make a copy of dataframe
    bg = entries.copy()
    
    # specify the date column name
    # the data column is not consistent
    date_col = "date"
    if bg["date"].dtype == int:
        date_col = "dateString"
    
    # extract the date and bg value (sgv)
    # remove duplicate entries due to error
    bg = bg[[date_col, "sgv"]] 
    bg = bg.drop_duplicates(ignore_index=True)
    bg = bg[bg[date_col].notnull()]
    
    # most systems are limited from 39 to 400 md/dl 
    # remove erroneous samples outside of this range
    bg = bg.loc[(bg["sgv"] <= 400) & (bg["sgv"] > 39.0)]
        
    # handle different string formats of date
    # process can be handled by pandas however it is much slower
    bg[date_col] = bg[date_col].apply(lambda row : modify_date(row))  
    bg[date_col] = pd.to_datetime(bg[date_col])  
    
    # round time to nearest minute and rename columns
    bg[date_col] = bg[date_col].dt.round('min')
    bg = bg.rename(columns={"sgv": "bg", date_col: "date"})

    return bg


def dict_to_array(row):
    """
    Background basal rates in profile are set as dictionary of 
    30-minute intervals. Convert them to array of basal rates.
    """

    array = np.zeros(int(24/0.5)) - 1   
    for entry in row:
        
        # get time of basal rate time application in seconds
        time_ob = time.strptime(entry["time"],'%H:%M')
        time_s = datetime.timedelta(
            hours=time_ob.tm_hour,
            minutes=time_ob.tm_min,
            seconds=time_ob.tm_sec
        ).total_seconds()
        
        # identify the corresponding 30-minute interval
        idx = int(time_s/(30 * 60))
        array[idx] = entry["value"]
    
    # forward fill the array
    for idx in range(1, len(array)):        
        if array[idx] == -1:
            array[idx] = array[idx - 1]
            
    return array


def convert_profile(profile, treatments):
    """
    Clean NightScout profile information and extract 
    the user set background basal doses.
    """

    # make a copy of dataframes
    prof = profile.copy()
    treat = treatments.copy()

    # ensure essential columns exist
    if "basal" not in prof.columns:
        prof["basal"] = np.nan        
    if "profileJson" not in treat.columns:
        treat["profileJson"] = np.nan    
    
    # process the profile file  -----------------------------------
    
    # exclude data without time information
    prof = prof.loc[prof["startDate"].notna()]
    
    # extract basal log data from column labelled "store"
    # contains a list of nested dictionaries
    prof_basal = [
        list(d.values())[0]["basal"] 
        for d in profile["store"] 
        if isinstance(d, dict)
    ]    
    
    # basal information can also be stored in "basal" variable
    # fill in empty basal column with informatuon from "store"
    # add a blank entry in case there are no recorded values
    prof_basal.append([{"test": None}])
    prof.loc[prof["basal"].isnull(), "basal"] = np.array(
        prof_basal, dtype=object)[:-1]
    
    # extract columns 
    prof = prof[[
        "startDate",  # datetime at which background basal was changed
        "basal"       # dict containing 48 30 min periods with basal
    ]]
    
    # add additional data from treatments ----------------------------------
    # basal information is not always recorded in profile on Nightscout
    # bolster information using events recorded by pump independently
    
    # extract relevant columns from treatment
    pf_s = treat[[
        "created_at",   # datetime when sample was created
        "eventType",    # tag given by insulin pump for event
        "profileJson"   # any information relating to profile changes
    ]]
    
    # identify logged profile switch events 
    pf_s = pf_s.loc[pf_s["profileJson"].notna()]
    pf_s = pf_s.loc[pf_s["eventType"] == "Profile Switch"] 
    pf_s["profileJson"] = pf_s["profileJson"].astype(str)    
    
    # remove duplicate notes referring to nighscout upload
    pf_s = pf_s.loc[~pf_s["profileJson"].str.contains(
        "info.nightscout.androidaps")]
    
    # filter profile column for basal defaults
    pf_s["profileJson"] = pf_s["profileJson"].apply(lambda x: ast.literal_eval(x)) 
    pf_s["profileJson"] = pf_s["profileJson"].apply(lambda x: x.get('basal'))
    
    # rename columns and filter
    pf_s = pf_s.rename(columns={"created_at": "startDate", "profileJson": "basal"}) 
    pf_s = pf_s[["startDate", "basal"]]
    pf_s["endDate"] = np.nan
    
    # combine the two streams ------------------------------------------
    
    # combine profile with info extracted from insulin pump
    prof = pd.concat([prof, pf_s])
    
    # standardise the date and round to per minute
    prof["startDate"] = prof["startDate"].apply(lambda row : modify_date(row)) 
    prof["startDate"] = pd.to_datetime(prof["startDate"])
    prof["startDate"] = prof["startDate"].dt.round('min')
    prof = prof.rename(columns={"startDate": "date"}) 
    
    # order chronologically 
    prof.sort_values(by='date', inplace=True)
    prof = prof.reset_index(drop=True)
    
    # add a seperate column which specifies the duration of profile activity
    # use 01-01-2022 to represent end date of most recent
    end_dates = [prof["date"].iloc[[idx + 1]].values[0] for idx in range(len(prof) - 1)]
    prof["endDate"] = end_dates + [datetime.datetime(2022, 1, 1)]   
    
    # set 01-01-1900 as the earliest start data
    prof = pd.concat([prof.iloc[[-1]], prof])    
    prof = prof.reset_index(drop=True) 
    prof.at[0, 'endDate'] = prof["date"].iloc[[1]].values[0]
    prof.at[0, 'date'] = datetime.datetime(1900, 1, 1)
        
    # restructure the dictionaries which originally store basal rates
    # into more workable arrays
    prof['basal'] = prof['basal'].apply(lambda row : dict_to_array(row))
        
    return prof


def get_background(row, prof):
    """
    Iterate through a dataframe and extract the background basal 
    dose delivered for that time of day based on insulin 
    delivery profile of user.
    """

    dt_obj, dt_64 = row.time(), row.to_datetime64()    
    time_idx = int((dt_obj.hour * 60 + dt_obj.minute) // 30)    

    # if timestamp of sample falls within start 
    # and end date of time insulin profile is active 
    back_basal = (prof.loc[(prof["date"] <= dt_64) & (prof["endDate"] > dt_64), "basal"].values[0])
    selected_dose = back_basal[time_idx] 
    
    return selected_dose 


def convert_basal(treatments, clean_profile, max_duration=24*60):
    """
    Extract basal dose information from the insulin pump 
    device log.   
    """    

    # copy dataframe
    basal = treatments.copy()
    
    # specify required processing columns
    # add blank entries if not present on specific device
    required_columns = [
        "absolute",          # absolute insulin dose delivered (units/hr)
        "percent",           # percentage of background basal dose delivered
        "notes",             # sometimes OpenAPS records events in this column
        "isSMB"              # super-micro boluses are used by 0ref1 
    ]
    for col in required_columns:
        if col not in basal.columns:
            basal[col] = np.nan
   
    # specify event tags relating to basal dosing
    # include only samples with these tags
    basal_events = [
        "Temp Basal",        # temporary changes to basal insulin over specified duration
        "OpenAPS Offline",   # Deactivation of OpenAPS will cause basal to revert to background
        "Correction Bolus",  # Super-micro-boluses can be classified as correction bolus
        "Note"               # Note can be used as a generic event type
    ]
    basal = basal[(treatments["eventType"].isin(basal_events))]            
    
    # specify columns relating to basal dosing
    basal_cols = [
        "created_at",   # timestamp
        "eventType",    # corresponding event tag
        "insulin",      # insulin dose (units/hr)
        "duration",     # time duration of temporary basal in minutes
        "notes",        # as above 
        "absolute",     #   ''
        "percent",      #   ''
        "isSMB"         #   ''
    ]
    basal = basal[basal_cols] 
    
    # remove exact copies
    basal = basal.drop_duplicates(ignore_index=True)
    
    # get correction bolus -------------------------------------------------------

    # add supermicro-boluses as basal doses 
    # these are controlled by 0ref1
    is_smb = (basal["isSMB"] == 1.0)
    basal["corr_bolus"] = basal.loc[is_smb, "insulin"]
    
    # set dose and duration to zero now correction is extracted 
    basal.loc[is_smb, "insulin"] = 0.0 
    basal.loc[is_smb, "duration"] = 0.0

    # fill empty entries of correction bolus with zero doses
    # fill "insulin" with insulin reported via "absolute" amount  
    basal["corr_bolus"] = basal["corr_bolus"].fillna(0.0)
    basal["insulin"] = basal["insulin"].fillna(basal["absolute"]) 
        
    # add pump start and stop events ---------------------------------------------  
  
    # want to stop insulin infusion when pump is paused
    # below are all the identified tags indicating this event
    stop_events = [
        "Suspend Pump", "Pump paused", "Pump stopped",
        "Pump Suspend", "Pump suspended", "PumpSuspend"
    ]    
        
    # when pump activity has stopped set insulin duration to zero
    # set suspend duration to a maximum duration of 24 hours
    for col in ["eventType", "notes"]:
        basal[col] = basal[col].fillna("NA")
        for ev in stop_events:
            basal.loc[basal[col].str.contains(ev), "insulin"] = 0.0
            basal.loc[basal[col].str.contains(ev), "duration"] = max_duration
    
    # re-start pump activity 
    start_events = [
        
        # when temporary basal dose has ended insulin 
        # defaults to the background insulin 
        "Basal Temp End", "Basal Temp Ende",
        
        # pump activity restarts
        "Pump resumed", "PumpResume", "Pump started", 
        "Resume Pump", "PumpResume", 
        
        # when OpenAPS algorithm goes offline 
        # defaults to the background insulin
        "OpenAPS Offline"
    ]
    
    # set duration of re-start events to zero
    # label insulin dose as -1 unit to indicate that this 
    # needs to be set to the background level
    for col in ["eventType", "notes"]:
        basal[col] = basal[col].fillna("NA")
        for ev in start_events:
            basal.loc[basal[col].str.contains(ev), "insulin"] = -1.0
            basal.loc[basal[col].str.contains(ev), "duration"] = 0.0
            
    # add background basal rates ---------------------------------------------
            
    # standardise the time for the recorded basal doses
    basal['created_at'] = basal['created_at'].apply(lambda row : modify_date(row))       
    basal["created_at"] = pd.to_datetime(basal["created_at"]).dt.tz_localize(None)            
    basal["created_at"] = basal["created_at"].dt.round('min')
    basal.sort_values(by='created_at', inplace=True)
    basal = basal.reset_index(drop=True)
        
    # set insulin of marked samples corresponding to resumption of 
    # pump activity to background insulin settings 
    act_times = basal.loc[basal["insulin"] == -1.0, "created_at"]
    act_back_basal = act_times.apply(lambda row : get_background(row, clean_profile)) 
    basal.loc[basal["insulin"] == -1.0, "insulin"] = act_back_basal 
       
    # some insulin doses are only specified as a percentage of the background basal
    # standardise these to be absolute values in units of insulin
    basal["back_insulin"] = basal["created_at"].apply(lambda row : get_background(row, clean_profile))
    basal["percent_insulin"] = (100 + basal["percent"])/100 * basal["back_insulin"]    
    basal["insulin"] = basal["insulin"].fillna(basal["percent_insulin"]) 
                  
    # rename the columns
    basal = basal[(basal["created_at"].notnull()) & (basal["insulin"].notnull())]
    basal = basal[["created_at", "insulin", "duration", "corr_bolus"]]
    basal = basal.rename(columns={"created_at": "date", "insulin": "basal"}) 
    
    # remove duplicate rows in super-micro-bolus doses
    # ensure multiple doses in 1 minute span are combined
    corr_basal = basal.loc[basal["corr_bolus"] > 0.0, ["date", "corr_bolus"]]
    corr_basal = corr_basal.drop_duplicates(['date', "corr_bolus"], ignore_index=True)     
    corr_basal = corr_basal.groupby(["date"], sort=False, as_index=False).sum() 

    # remove duplicate tempoaray basal doses
    # combine the SMB doses with temporary and background basal events
    temp_basal = basal.loc[basal["duration"] != 0, ["date", "basal", "duration"]]
    temp_basal = temp_basal.drop_duplicates(['date'], keep='last', ignore_index=True)   
    basal = reduce(lambda left, right: pd.merge(
        left, right, on=['date'], how='outer'), [temp_basal, corr_basal]) 
    
    # ensure there are no duplicate samples in the data
    assert len(basal.loc[basal.duplicated()]) == 0, "Duplicates exist in basal."
        
    return basal


def convert_bolus(treatments):
    """
    Extract bolus and carbohydrate information from the 
    insulin pump logs, including combination boluses.
    """

    # duplicate the original dataframe
    bolus = treatments.copy()    
    
    # ensure the following columns are in the dataframe
    required_columns = [
        "enteredinsulin",      # insulin dose typically associated with combo bolus
        "relative",            # used for combo boluses to specify total prolonged insulin 
        "isSMB"                # super-micro boluses are used by 0ref1 
    ]
    for col in required_columns:
        if col not in bolus.columns:
            bolus[col] = np.nan
    
    # filter bolus specific events
    bolus_events = [
        "Carb Correction",      # insulin taken post meal to compensate for carbohydrates
        "Meal Bolus",           # insulin taken for a meal
        "Snack Bolus",          # a small insulin dose to compensate a snack
        "Bolus Wizard",         # another meal bolus tag, associated with carbohydrate input 
        "Combo Bolus",          # a dual bolus with immediate and prolonged insulin
        "Note",                 # insulin doses sometimes associated with this tag
        "<none>",               # insulin doses sometimes associated with this tag
        "Carbs",                # user manually enters carbohydrate in grams for meal
        "Bolus",                # a large dose of insulin typically in relation to meal
        "Correction Bolus"      # insulin typically taken to correct for high blood glucose
    ]    
    bolus = bolus[bolus["eventType"].isin(bolus_events)] 
            
    # filter the bolus specific columns
    bolus_cols = [
        "created_at",      # timestamp of sample
        "eventType",       # nature of event
        "insulin",         # dose of insulin in units
        "carbs",           # carbohydrates consumed in grams
        "duration",        # time over which combo bolus is active
        "enteredinsulin",  # as above 
        "relative",        #   ''
        "isSMB"            #   ''
    ]
    bolus = bolus[bolus_cols]
    
    # filter correction boluses ----------------------------------------------
    
    # filter out super-micro-boluses
    # these are included with basal doses
    not_smb = (bolus["isSMB"] != 1.0)
    bolus = bolus.loc[not_smb]
            
    # standardise the time for the recorded bolus doses
    bolus = bolus[bolus["created_at"].notnull()]    
    bolus['created_at'] = bolus['created_at'].apply(lambda row : modify_date(row))   
    bolus["created_at"] = pd.to_datetime(bolus["created_at"]).dt.tz_localize(None)    
    bolus["created_at"] = bolus["created_at"].dt.round('min')
    
    # remove duplicate entries
    # fill samples without carbohydrates or insulin with zero
    bolus = bolus.reset_index(drop=True)
    bolus["carbs"] = bolus["carbs"].fillna(0.0)
    bolus["insulin"] = bolus["insulin"].fillna(0.0)
    bolus = bolus.drop_duplicates(["created_at", "carbs", "insulin"], ignore_index=True)
    
    # add combo boluses -----------------------------------------------------
        
    # identify combination boluses
    # combo bolus should have: 
    # - duration -> duration of prolonged bolus in mins
    # - relative insulin dose -> insulin given in prolonged bolus (units)
    # - insulin ->  insulin given immediately (units)
    combo_bolus = bolus.loc[bolus["eventType"] == "Combo Bolus"]
    is_ins = (combo_bolus["insulin"].notna()) 
    is_unnorm_ins = (combo_bolus["enteredinsulin"].notna()) | (combo_bolus["relative"].notna())
    is_duration = (combo_bolus["duration"].notna())
    combo_bolus = combo_bolus.loc[
        ((is_ins) | (is_unnorm_ins)) & (is_duration)
    ]
    
    # calculate the numbe of 5 minute intervals covered
    # calculate square bolus dose per interval
    combo_bolus["steps"] = combo_bolus["duration"]/5.0    
    combo_bolus["insulin"] = combo_bolus["insulin"].fillna(0.0)
    combo_bolus["relative"] = combo_bolus["relative"].fillna(0.0)
    combo_bolus["square_bolus"] = (combo_bolus["relative"])/combo_bolus["steps"] 
    
    # remove combo from bolus
    # to avoid duplicating doses
    bolus = bolus.loc[bolus["eventType"] != "Combo Bolus"]
                   
    # create an entry for each 5-minute period of the square bolus
    new_bolus = []
    for _, row in combo_bolus.iterrows():
        
        # create bolus range 
        dt = pd.Timedelta(minutes=5*row["steps"])
        combo_range = pd.date_range(
            start=row["created_at"], end=row["created_at"]+dt, freq='5min'
        ).to_frame(index=False, name="created_at")
        
        # set insulin and log
        combo_range["insulin"] = row["square_bolus"] 
        new_bolus.append(combo_range)
        
    # combine and filter -------------------------------------------
    
    # merge the added combination boluses with the 
    # original combination bolus information
    combo_bolus = pd.concat([combo_bolus] + new_bolus)
    
    # merge combination bolus events with the
    # original bolus information
    # sum insulin doses within the same interval
    combo_bolus = combo_bolus[["created_at", "insulin", "carbs"]]
    bolus = bolus[["created_at", "insulin", "carbs"]]    
    bolus = pd.concat([bolus, combo_bolus]) 
    bolus = bolus.groupby(["created_at"], sort=False, as_index=False).sum() 
    
    # fill unspecified bolus doses with zero
    # only include samples with bolus insulin or carbohydrates
    bolus = bolus.fillna(0.0)
    bolus = bolus.loc[(bolus["insulin"] != 0.0) | (bolus["carbs"] != 0.0)]
    bolus = bolus.rename(columns={"created_at": "date", "insulin": "bolus"}) 

    # ensure no duplicates exist within the data
    assert len(bolus.loc[bolus.duplicated()]) == 0, "Duplicates exist in bolus."

    return bolus


def convert_treatments(treaments, clean_profile, max_duration=24*60):
    """
    Process the insulin pump device logs. Profile information is 
    required to add background basal rates.  
    """

    # copy the data
    treat = treaments.copy()
    
    # extract basal and basal information
    basal = convert_basal(treat, clean_profile, max_duration=max_duration)
    bolus = convert_bolus(treat)
    
    return basal, bolus
        
    
def merge_data_streams(bg, basal, bolus, clean_profile, debug=0):
    """
    Combine basal, bolus and blood glucose information 
    into a continuous stream of data occuring at 5-minute intervals. 
    """

    # merge the dataframes -----------------------------------------------------
        
    # combine blood glucose, basal and bolus information
    data_frames = [bg, basal, bolus]        
    combined = reduce(lambda left, right: pd.merge(
        left, right, on=['date'], how='outer'), data_frames) 
    
    # round the data points to nearest 5-minute interval
    combined.sort_values(by='date', inplace=True)
    combined["date"] = combined["date"].dt.round('5min')
    combined = combined.reset_index(drop=True)
        
    # sum bolus and carbohydrate doses within each interval
    combined_sum = combined[
        ["date", "bolus", "corr_bolus", "carbs"]].copy().groupby("date").sum() 
    
    # keep the most recent blood glucose measurement in each interval
    last_combined = combined[["date", "bg"]].copy().drop_duplicates(
        ['date'], keep='last', ignore_index=True)    

    # recombine the blood glucose and bolus/carbohydrate information
    matched_bolus = pd.merge(combined_sum, last_combined, on=["date"], how='inner')
    matched_bolus = matched_bolus.drop_duplicates(['date'], ignore_index=True)
    combined.sort_values(by='date', inplace=True)
    combined = combined.reset_index(drop=True)
    
    # fill in gaps in dataframe ------------------------------------------------------
    
    # create additional samples at 5-minute intervals over the full dataframe
    max_time, min_time = combined["date"].max(), combined["date"].min()
    missing_rows = pd.date_range(
        start=min_time, end=max_time+pd.Timedelta(minutes=5), freq='5min'
    ).to_frame(index=False, name="date")    
    
    # label newly created rows to distinguish from original
    missing_rows["not_missing"] = 0.0
    combined["not_missing"] = 1.0    
    
    # add blank samples and keep those that already exist in data
    combined = pd.concat([combined, missing_rows])
    combined = combined.groupby("date", as_index=False).sum(min_count=1)
    combined["not_missing"] = combined["not_missing"].clip(upper=1.0)
    
    # create intermediate basal entries -------------------------------------------------
    
    # get the most recent basal rate and duration
    basal_rel = combined[["date", "bg", "basal", "duration"]].copy()
    basal_rel = basal_rel.drop_duplicates(
        ['date'], keep='last', ignore_index=True)
    
    # get the current temporary basal duration and date
    # also get the background basal rate corresponding to earliest sample
    first_row = basal_rel.iloc[[0]]        
    prev_time, cur_duration = first_row["date"].values[0], 1
    cur_rate = first_row["date"].apply(
        lambda row : get_background(row, clean_profile)).values[0]   

    # iterate through the samples and label the added blank samples
    # with correct basal dose. If temporary basal is active set
    # insulin dose to that. Otherwise check background basal rate
    # or if insulin pump is currently functioning
    basal_log = [cur_rate]    
    for idx, row in enumerate(basal_rel.to_numpy()[1:, :]):
        if (idx % 10_000 == 0) and (idx > 0) and (debug > 1):
            print(f'Processed: {idx}/{len(basal_rel)}')
                
        # if basal dose and basal duration exist
        # update the current rate and duration
        if (not pd.isnull(row[2])) and (not pd.isnull(row[3])): 
            cur_duration = int(row[3]//5)
            cur_rate = row[2]
        
        # if duration is has run out on previous temporary basal
        # set current rate to the background basal rate
        if cur_duration <= 0:      
            current_row = basal_rel.iloc[[idx + 1]]  
            cur_rate = current_row["date"].apply(
                lambda row : get_background(row, clean_profile)
            ).values[0] 
        
        # update the remaining duration
        time_diff = (row[0] - prev_time).total_seconds()/(60*5)        
        cur_duration -= time_diff 
                    
        # log the basal for the timestep
        basal_log.append(cur_rate) 
        prev_time = row[0]

    # set the basal dose with iterated basal values
    combined["basal"] = basal_log
    
    # combine the updated basal information with the bolus events
    final_df = pd.merge(combined[[
        "date", "basal", "not_missing"]], matched_bolus, on=["date"], how='outer')
    final_df = final_df[["date", "bg", "basal", "carbs", "bolus", "corr_bolus", "not_missing"]].copy() 
        
    # fill in final gaps -------------------------------------------------
    
    # fill the empty entries in bolus, basal and carbs
    final_df["bolus"] = final_df["bolus"].fillna(0.0)
    final_df["carbs"] = final_df["carbs"].fillna(0.0)
    final_df["corr_bolus"] = final_df["corr_bolus"].fillna(0.0)
    final_df.sort_values("date", ignore_index=False)
    final_df = final_df.reset_index(drop=True)
    
    # set minumum to zero for bolus, basal, carbs and corr_bolus
    final_df[["basal", "carbs", "bolus"]] = final_df[
        ["basal", "carbs", "bolus"]].clip(lower=0.0)

    # convert basal to rate (units per hr) to absolute (units)
    # combine correction bolus with basal
    final_df["basal"] = (final_df["basal"] * 5)/60 + final_df["corr_bolus"]
    final_df = final_df[[
        "date",            # timestamp rounded to nearest 5 minutes
        "bg",              # blood glucose measurement from CGM in mg/dl
        "basal",           # basal insulin dose for 5-minute interval (units)
        "carbs",           # carbohydrates consumed in 5-minute interval (grams)
        "bolus",           # bolus insulin given in internval (units)
        "not_missing"      # tag indicating if sample was added or original (if not_missing == "original")
    ]].copy()
        

    return final_df