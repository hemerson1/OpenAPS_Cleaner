"""
Re-usable functions for dataset pre-processing.
"""


def read_selected_pts(pt_file=None):
    """
    Select participants from the OpenAPS repository.
    Due to non-standardised nature of the dataset not 
    all participants will have data in the correct format
    or with compatible devices.   
    """
    
    # use a default file
    if pt_file is None: pt_file = './datasets/pts.txt'

    # read file and convert to list
    with open(pt_file, "r") as file:
        pts = file.read().splitlines()
    return pts