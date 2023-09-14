"""
Process the supplementary demographic information 
with the OpenAPS Data Commons
"""

import numpy as np
import pandas as pd

from cleaner.helper import *

def load_demographic_data(path, pt_file=None):
    """
    Load the OpenAPS demographic file for the patients in Repository.
    Standardise the information in participant answered a questionnaire.    
    """

    # load excel demographic file and extract the relevant columns
    df = pd.read_excel(path, sheet_name=None)     
    relevant_qs = [
        'Your OpenHumans OpenAPS Data Commons "project member ID"',
        "When were you born?",
        "What type of DIY close loop technology do you use? Select all that you actively use:",
        "How much do you weigh?",
        "How many units of insulin do you take per day?",
        "On average, how many grams of carbohydrates do you eat in a day?",
        'Timestamp',
        "How tall are you?",
        "Gender",
        "What was your last lab-reported A1C?" 
    ]

    # filter by pre-specificed columns and make more concide
    df = df['Form Responses 1'][relevant_qs]
    df = df.rename(
        columns={
            relevant_qs[0]: "PtID",
            relevant_qs[1]: "age",
            relevant_qs[2]: "system",
            relevant_qs[3]: "weight",
            relevant_qs[4]: "TDI", 
            relevant_qs[5]: "daily_carbs", 
            relevant_qs[6]: "current_date", 
            relevant_qs[7]: "height", 
            relevant_qs[8]: "gender",
            relevant_qs[9]: "mean_bg",
        }
    )
    
    # ensure patient ID is consistent length
    df["PtID"] = df["PtID"].astype(str)
    df['PtID'] = df['PtID'].str.zfill(8)

    # filter out selected participants
    pt_list = read_selected_pts(pt_file=pt_file)
    df = df.loc[df["PtID"].isin(pt_list)]
    
    # users accidentally start form but do not finish
    # keep only most recent results
    df.sort_values(["PtID", "age"])
    df = df.drop_duplicates(subset="PtID", keep="last")

    # convert birth date to age at time of filling out form
    df['age'] = pd.to_datetime(df['age'])
    df["current_date"] = pd.to_datetime(df['current_date'])
    df["age"] = (df["current_date"] - df["age"]).dt.days / 365.25

    # label any user which uses OpenAPS algorithm at any point
    df = df.loc[df["system"].str.contains("OpenAPS|AndroidAPS")]
    df.loc[df["system"].str.contains("OpenAPS"), "system"] = "OpenAPS" 

    # convert weight to kg
    df['weight'] = df['weight'].astype(str).str[0:3]
    df['weight'] = pd.to_numeric(df['weight'])
    df['weight'] = np.where(

        # it is challenging to identify if patient weight is in kg
        # assumed weights listed as >120 are likely to be in lbs
        # however for child aged partipants it is hard to tell
        np.logical_or((df['weight'] > 120), (df['weight'] < 60)),
        df['weight'] * 0.453592, # (kg/lb)
        df['weight']
    )
    
    # standardise the notation of the column
    # some users specify a range (e.g. 10 - 20 units) so take the mean
    df['TDI'] = df['TDI'].astype(str).str.replace(',', '.', regex=True)    
    df['TDI'] = df['TDI'].str.split('-', expand=True).astype(float).mean(axis=1)
    
    # convert carbs to standard form   
    # # some users specify a range (e.g. 100 - 200g) so take the mean 
    split_carbs = df['daily_carbs'].astype(str).str.split('-', expand=True)
    df['daily_carbs'] = split_carbs.astype(float).mean(axis=1)
       
    # convert height in feet to cms 
    # extract 3 digits from each entry
    # assume first to be feet and last two to be inches
    df["height"] = df["height"].astype(str)
    height_ft = df.copy()
    height_ft["height"] = height_ft["height"].str.replace(
        '\D+', '', regex=True).str.slice(0, 3)

    # three continuous numbers are likely to already be in cms
    df["height"] = df["height"].str.extract('(\d{3})')
    
    # convert feet to cms
    height_ft['feet'] = height_ft['height'].str[0]
    height_ft["feet"] = height_ft['feet'].astype(float) * 30.48 # (cm/ft)

    # convert inches to cms and add zero if no second set of digits 
    height_ft['inches'] = height_ft['height'].str[1:3]
    height_ft['inches'] = height_ft['inches'].apply(lambda x: "0" if x == '' else x)    
    height_ft["inches"] = height_ft['inches'].astype(float) * 2.54 # (cm/inch)

    # inches + feet and replace missing entries  
    height_ft["height_cm"] = height_ft["feet"] + height_ft["inches"]    
    df["height"] = df["height"].fillna(height_ft["height_cm"])

    # fill-in unreported gender
    df["gender"] = df["gender"].fillna("N/A")
    
    # ensure mean blood glucose is a percentage
    # A1C(%) = 10.929 * (A1C(mmol/mol) - 2.15)
    df['mean_bg'] = df['mean_bg'].astype(str).str.replace(',', '.', regex=True)
    df["mean_bg"] = df["mean_bg"].str.extract('(\d+\.?\d*)')
    df["mean_bg"] = df["mean_bg"].astype(float)
    unit_bg = df.loc[df["mean_bg"] > 18, "mean_bg"]
    df.loc[df["mean_bg"] > 18, "mean_bg"] = unit_bg / 10.929 + 2.15 
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    
    DEMO_PATH = './datasets/OpenAPS_Demographic.xlsx'
    demo = load_demographic_data(path=DEMO_PATH)
    print(demo)