


demo_path = './datasets/OpenAPS_Demographic.xlsx'
dataset_path = './datasets/Processed_data/test_data'
save_path = './datasets/Processed_data/test_segments'
pt_path = './datasets/pts.txt'
num_jobs = 8

params={

    # state representation
    "state_type": "default",
    "reward_type": "magni",
    
    # processing
    "traj_len": 8, # hrs
    "time_diff_thresh": 30, # mins

    # augmentation
    "augment_options": [],
    "filter_options": [],
    "sensor_options": []
}
    