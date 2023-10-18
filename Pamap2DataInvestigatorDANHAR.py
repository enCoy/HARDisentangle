# same as BPD except for downsampling by 3

import numpy as np
import pandas as pd
import os
import sys
from scipy.io import savemat
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from scipy.signal import decimate, resample

def get_heartrate_corrections_pamap2():
    # values are tuples (number of nans at the beginning, first value after the nans)
    # keys are subject ids
    dict_to_use = {
        1: (4, 100),
        2: (5, 96),
        3: (6, 91),
        4: (9, 106),
        5: (5, 98),
        6: (6, 103),
        7: (5, 80),
        8: (0, 84)
    }
    return dict_to_use
def get_pamap2_activities():
    activityIDdict = {0: 'transient',
                      1: 'lying',  # used
                      2: 'sitting',  # used
                      3: 'standing',  # used
                      4: 'walking',  # used
                      5: 'running',  # used
                      6: 'cycling',  # used
                      7: 'Nordic_walking',  # used
                      9: 'watching_TV',
                      10: 'computer_work',
                      11: 'car driving',
                      12: 'ascending_stairs',  # used
                      13: 'descending_stairs',  # used
                      16: 'vacuum_cleaning',   # used
                      17: 'ironing',  # used
                      18: 'folding_laundry',
                      19: 'house_cleaning',
                      20: 'playing_soccer',
                      24: 'rope_jumping'  # used
                      }
    return activityIDdict

def label_correcter():
    label_mapping = {1: 0, # 'lying'
                     2: 1,  # sitting
                     3: 2,  # standing
                     4: 3,  # walking
                     5: 4,  # running
                     6: 5,  # cycling
                     7: 6,  # nordic walking
                     12: 7,  # ascending stairs
                     13: 8,  # descending stairs
                     16: 9,  # vacuum cleaning
                     17: 10,  # ironing
                     24: 11,  # rope jumping
                      }
    return label_mapping

def get_pamap2_column_names():
    colNames = ["timestamp", "activityID", "heartrate"]

    IMUhand = ['handTemperature',
               'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
               'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
               'handGyro1', 'handGyro2', 'handGyro3',
               'handMagne1', 'handMagne2', 'handMagne3',
               'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

    IMUchest = ['chestTemperature',
                'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
                'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
                'chestGyro1', 'chestGyro2', 'chestGyro3',
                'chestMagne1', 'chestMagne2', 'chestMagne3',
                'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

    IMUankle = ['ankleTemperature',
                'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
                'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

    columns = colNames + IMUhand + IMUchest + IMUankle  # all columns in one list
    return columns

def clean_pamap2_data(data):
    # source: https://github.com/andreasKyratzis/PAMAP2-Physical-Activity-Monitoring-Data-Analysis-and-ML/blob/master/pamap2.ipynb
    processed_data = data.drop(data[data.activityID == 0].index)  # removal of any row of activity 0 as it is transient activity which it is not used
    processed_data = processed_data.apply(pd.to_numeric)  # removal of non numeric data in cells
    processed_data = processed_data.interpolate()  # removal of any remaining NaN value cells by constructing new data points in known set of data points
    return processed_data


if __name__ == "__main__":
    project_dir = r"C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR"
    dataset_dir = os.path.join(project_dir, 'PAMAP2_Dataset\PAMAP2_Dataset\Protocol')
    outpur_dir = os.path.join(project_dir, 'PAMAP2_Dataset\PAMAP2_Dataset\Processed50Hz')
    subject_ids_to_process = [1, 2, 3, 4, 5, 6, 7, 8]

    for subject_id in subject_ids_to_process:
        df = pd.read_table(os.path.join(dataset_dir, f'subject10{subject_id}.dat'), header=None, sep='\s+')
        df.columns = get_pamap2_column_names()
        processed_data = clean_pamap2_data(df)
        processed_data.reset_index(drop=True, inplace=True)
        # heart rate correction - first few samples are still nan after interpolation
        print(processed_data.head())
        heart_rate_correction_map = get_heartrate_corrections_pamap2()
        for i in range(0, heart_rate_correction_map[subject_id][0]):
            processed_data["heartrate"].iloc[i] = heart_rate_correction_map[subject_id][1]
        print("processed data shape: ", processed_data.values.shape)
        labels = processed_data['activityID'].values
        # process labels
        label_mapping = label_correcter()
        new_labels = np.zeros_like(labels)
        for j in range(labels.shape[0]):
            new_labels[j] = label_mapping[labels[j]]


        data = processed_data.drop(['timestamp', 'activityID'], axis=1).values
        print("data shape: ", data.shape)
        print("label shape: ", new_labels.shape)
        downsampled_data = decimate(data, q=2, axis=0)  # 100 hz to 50 hz
        downsampled_label = new_labels[::2]
        print("downsampled data shape: ", downsampled_data.shape)
        print("downsampled label shape: ", downsampled_label.shape)


        mdic = {'data': downsampled_data,
                'label': downsampled_label}
        # save the data
        savemat(os.path.join(outpur_dir, f"subject10{subject_id}.mat"), mdic)

        with open(os.path.join(outpur_dir, f"subject10{subject_id}.pickle"), 'wb') as handle:
            pickle.dump(mdic, handle, protocol=pickle.HIGHEST_PROTOCOL)
