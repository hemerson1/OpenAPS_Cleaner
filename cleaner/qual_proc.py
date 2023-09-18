'''
Convert the qualitative notes left by patients into tags 
of blood glucose events.
'''

import warnings
import numpy as np
import pandas as pd

from cleaner.helper import *

# hide pandas warning
warnings.filterwarnings(
    "ignore", 'This pattern is interpreted as a regular expression')


def process_event_msgs(treatments):
    """
    Process user reported qualitative tags into binary
    indicators of a given event. Meal composition was 
    approximated with reference to: https://fdc.nal.usda.gov/

    Includes: exercise (low/high intensity), high fat meals,
    high protein meals, caffeine, alcohol

    NOTE: labelling events was performed manually and therefore 
    scaling to larger datasets would likely require an automated 
    method. This was have to consider a wide variety of items 
    across multiple different languages. 
    """

    # duplicate the original dataframe
    treat = treatments.copy()
    
    # extract the event columns
    if "notes" not in treat.columns: treat["notes"] = np.nan
    event_log = treat[[
        "created_at",         # timestamp of sample
        "eventType",          # description of recorded treatment event
        "notes"               # event information can often be recorded in notes
    ]]
    
    # standardise the sample date and time
    event_log = event_log[event_log["created_at"].notnull()]    
    event_log['created_at'] = event_log['created_at'].apply(lambda row : modify_date(row))   
    event_log["created_at"] = pd.to_datetime(event_log["created_at"]).dt.tz_localize(None)    
    event_log["created_at"] = event_log["created_at"].dt.round('5min')
    
    # convert event type column to string
    # fill empty records with "N/A"
    event_log["eventType"] =  event_log["eventType"].astype(str)
    event_log["notes"] =  event_log["notes"].astype(str)
    event_log["notes"] = event_log["notes"].str.lower()
    event_log.sort_values("created_at", ignore_index=False)
    event_log = event_log.rename(columns={"created_at": "date"}) 
    event_log = event_log.reset_index(drop=True)
    event_log = event_log.fillna("N/A")
    
    # Physical Activity ----------------------------------------------------

    # create a column for exercise events
    # when exercise tag provided label as such
    event_log["exercise"] = 0
    event_log.loc[event_log["eventType"] == "Exercise", "exercise"] = 1
    
    # high intensity
    high_exercise = [
        "fahrrad", "fahren", "biking", "rad",  # cycling
        "lauf", "run",                         # running
        "climbing",                            # climbing
        "schwim",                              # swimming
        "exercise", "sport", "trännng"         # generic
    ]

    # low intensity (walking)
    low_exercise = [
        "spaziergang", "walk", "gehen",
        "wandern"
    ]

    # label the exercise events accordingly
    exercise_notes = high_exercise + low_exercise
    is_exercise = event_log["notes"].str.contains(
        "|".join(exercise_notes))    
    event_log.loc[is_exercise, "exercise"] = 1 
    is_high_exercise = event_log["notes"].str.contains(
        "|".join(high_exercise))    
    event_log.loc[is_high_exercise, "high_exercise"] = 1 
    is_low_exercise = event_log["notes"].str.contains(
        "|".join(low_exercise))    
    event_log.loc[is_low_exercise, "low_exercise"] = 1 

    # Fatty meals ---------------------------------------------------
    
    # add potential high fat food labels 
    event_log["high_fat"] = 0
    high_fat_meals = [
        
        # sweet
        "waffeln", "kinder", "kaba", "tiramisu", 
        "schok", "duplo", "snicker", "kuchen", 
        "kakao", "choco", "muffin", "rollos", 
        "cookie", "koko", "keks", "knopper",
        
        # high carb
        "raviolli", "gnocchi", "tortellini", "farfalle",
        "pizza", "curry", "fritt", "chips", "gyro",
        "spaghetti", "kartoffel", "lasagne", "nudeln", 
        "pasta", "döner", "knödel", "pommes", 
        "croissant", "lizza", "potato", "nudelsalat",
        
        # dairy
        "camembert", "cheese", "milch", "eis", 
        "butter", "buttee", "mjölk", "eier", 
        "joghurt", "creme", "cream", "kreme",
        "käse", "yoghurt", "mozzarella", "glass",
        "feta"
        
        # misc.
        "kfc", "mayonnaise", "brisket", "fried",
        "hollandaise", "duck", "mcdonalds", "burger",
        "steak", "skinka"
    ]
    
    # label corresponding events
    is_high_fat = event_log["notes"].str.contains(
        "|".join(high_fat_meals))
    event_log.loc[is_high_fat, "high_fat"] = 1 

    # High protein ----------------------------------------------------------
    
    # add high protein meals
    event_log["high_protein"] = 0
    protein_meals = [
    
        # meat
        "chicken", "fleisch", "salsiccia",
        "meat", "döner", "chorizo", "wurst",
        "burger", "schwein", "gyro", "würst",
        "brisket", "steak", "skinka", "duck", 
        "hünchen", "schnitzel", "shrimp", "meatball", 
        "salami",
        
        # dairy
        "quark", "cheese", "feta", "milch", 
        "eiweiß", "camembert", "eier", "mjölk",
        "mozzarella", "käse",
        
        # misc.
        "protein"
    ]

    # label corresponding events
    is_high_protein = event_log["notes"].str.contains(
         "|".join(protein_meals))
    event_log.loc[is_high_protein, "high_protein"] = 1 

    # Caffeine ---------------------------------------------------------------
    
    # add caffeine meals
    event_log["caffeine"] = 0
    caffeine_meals = [
        "kaffee", "cappuccino", "red bull",
        "coffee", "mountain dew", "redbull", 
        "cola", "espresso", "latte", "macchiato"
    ]

    # label corresponding events
    is_caffeine = event_log["notes"].str.contains("|".join(caffeine_meals))
    event_log.loc[is_caffeine, "caffeine"] = 1

    # Alcohol ----------------------------------------------------------------
    
    # add alcohol
    event_log["alcohol"] = 0
    alcohol_event = [
        "alkohol", "(^|[ ])alk", "aperol",
        "lagune", "(^|[ ])gin", "wein", "bier",
        "cocktsils", "schorle", "cocktails"
    ]

    # label corresponding events
    is_alcohol = event_log["notes"].str.contains(
        "|".join(alcohol_event))
    alcohol_free = event_log["notes"].str.contains(
        "|".join(["free", "frei"]))
    event_log.loc[(is_alcohol) & (~alcohol_free), "alcohol"] = 1
    
    return event_log
    