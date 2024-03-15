import numpy as np
import pandas as pd
import os

folders_to_activities = {
    'dws_1': 0,    'dws_11': 0,    'dws_2': 0,
    'jog_16': 1,    'jog_9': 1,
    'sit_13': 2,     'sit_5': 2,
    'std_14': 3,    'std_6': 3,
    'ups_12': 4,    'ups_3': 4,    'ups_4': 4,
    'wlk_15': 5,     'wlk_7': 5,    'wlk_8': 5
}

if __name__ == "__main__":

    dataset_dir = r"C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR\MotionSenseDataset"
    general_data_path = os.path.join(dataset_dir, 'A_DeviceMotion_data')
    num_subjects = 24
    subtract_gravity = False
    dataframe_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'activity_id']
    output_dir = os.path.join(dataset_dir, 'SubjectWiseData')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for j in range(1, num_subjects + 1):  # traverse subjects
        print(f"Subject {j} is being processed!")
        subject_whole_data = np.empty((0, 7))  # 7 = 3 acc + 3 gyr + 1 activity_label
        folders = os.listdir(general_data_path)
        for folder in folders:  # traverse activity folders
            data_path = os.path.join(general_data_path, folder, f'sub_{j}.csv')
            data = pd.read_csv(data_path)

            gravity__data = data[['gravity.x', 'gravity.y', 'gravity.z']]
            acc_data = data[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].values
            gyr_data = data[['rotationRate.x', 'rotationRate.y', 'rotationRate.z']].values
            activity_data = np.ones((len(acc_data), 1)) * folders_to_activities[folder]
            if subtract_gravity:
                acc_data = acc_data - gravity__data
            concatted = np.concatenate((acc_data, gyr_data, activity_data), axis=1)
            subject_whole_data = np.concatenate((subject_whole_data, concatted), axis=0)

        subject_dataframe = pd.DataFrame(data=subject_whole_data, columns=dataframe_columns)
        subject_dataframe.to_csv(os.path.join(output_dir, f'subj_{j}.csv'))


