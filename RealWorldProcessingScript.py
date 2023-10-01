import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
# suppress scientific notation by setting float_format
pd.options.display.float_format = '{:.0f}'.format
# activities = climbing stairs down and up, jumping, lying, standing, sitting, running/jogging,
SAMPLING_PERIOD = 20# 50 hz sampling = 20 ms sampling period

def custom_linspace(start, stop, step):
        """
          Like np.linspace but uses step instead of num
          This is inclusive to stop, so if start=1, stop=3, step=0.5
          Output is: array([1., 1.5, 2., 2.5, 3.])
        """
        return np.linspace(start, stop, int((stop - start) / step + 1))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_synchronization_time_array(data_dir, activity, sensors, locations):
    # get the maximum starting time of the devices and minimum of ending times - we will cut from both ends
    max_starting_time = -np.inf
    min_ending_time = np.inf
    time_array = None
    for sensor in sensors:
        for location in locations:
            # read the zip
            activity_zip = ZipFile(os.path.join(data_dir, f'{sensor}_{activity}_csv.zip'))
            # read csv
            if sensor == 'gyr':
                df = pd.read_csv(activity_zip.open(f'Gyroscope_{activity}_{location}.csv'))
            else:
                df = pd.read_csv(activity_zip.open(f'{sensor}_{activity}_{location}.csv'))

            activity_start_time = df['attr_time'].values[0]
            activity_end_time = df['attr_time'].values[-1]
            if activity_end_time < min_ending_time:
                min_ending_time = activity_end_time
            if activity_start_time > max_starting_time:
                max_starting_time = activity_start_time

            # create a time array
            time_array = custom_linspace(max_starting_time,
                                         min_ending_time,
                                         step=SAMPLING_PERIOD)
    return time_array, max_starting_time

def get_nn_indices_based_on_time(source_time, target_time):
    # in target time data, we are going to choose the indices that are closest to the ones in source time
    indices = []
    for current_time in source_time:
        idx, _ = find_nearest(target_time, current_time)
        indices.append(idx)
    return indices

def process_data(data_dir, subject_id, device_locations, sensors):
    # for each subject and activity create a single data frame - combine sensors in columns
    data_directory = os.path.join(data_dir, 'Raw', f'proband{subject_id}', 'data')
    for activity in activities:
        print("Subject: ", subject)
        print("Activity: ", activity)
        activity_df = pd.DataFrame()

        time_array, baseline_time = get_synchronization_time_array(data_directory, activity, sensors, device_locations)
        for sensor in sensors:  # [acc, gyro]
            print("Sensor:", sensor)
            activity_zip = ZipFile(os.path.join(data_directory, f'{sensor}_{activity}_csv.zip'))
            for location in device_locations:
                print("Location: ", location)
                if sensor == 'gyr':
                    df = pd.read_csv(activity_zip.open( f'Gyroscope_{activity}_{location}.csv'))
                else:
                    df = pd.read_csv(activity_zip.open( f'{sensor}_{activity}_{location}.csv'))


                time_indices = get_nn_indices_based_on_time(time_array, df[['attr_time']].values)
                data_to_concat = df[['attr_time', 'attr_x', 'attr_y', 'attr_z']].values
                activity_df[[f'{location}_{sensor}_t',
                            f'{location}_{sensor}_x',
                            f'{location}_{sensor}_y',
                            f'{location}_{sensor}_z']] = data_to_concat[time_indices]
                # subtract baseline time
                activity_df[f'{location}_{sensor}_t'] = activity_df[f'{location}_{sensor}_t'] - baseline_time
                activity_df['activity_id'] = np.ones(len(time_indices)) * activity_name_to_class[activity]
            print()
        output_dir = os.path.join(data_dir, 'Processed', f'subject{subject_id}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        activity_df.to_csv(os.path.join(output_dir, f'{activity}.csv'), float_format='%.10f')
    # now concatenate all activity dataframes


if __name__ == "__main__":
    dataset_dir = r"C:\Users\Cem Okan\Dropbox (GaTech)\DisentangledHAR\realworld2016_dataset"
    output_dir = os.path.join(dataset_dir, 'Processed')


    subject = 14

    if (((subject == 4) or (subject == 7)) or subject==14):
        activity_name_to_class = {
            'climbingdown_1': 0,
            'climbingdown_2': 0,
            'climbingdown_3': 0,
            'climbingup_1': 1,
            'climbingup_2': 1,
            'climbingup_3': 1,
            'jumping': 2,
            'lying': 3,
            'running': 4,
            'sitting': 5,
            'standing': 6,
            'walking': 7
        }
    else:
        activity_name_to_class = {
            'climbingdown': 0,
            'climbingup': 1,
            'jumping': 2,
            'lying': 3,
            'running': 4,
            'sitting': 5,
            'standing': 6,
            'walking': 7
        }


    activities = list(activity_name_to_class.keys())
    print("activities: ", activities)
    body_device_locations = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']


    sensors = ['acc', 'gyr']
    process_data(dataset_dir, subject, body_device_locations, sensors)

