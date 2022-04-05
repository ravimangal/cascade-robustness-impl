import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from functools import reduce
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MinMaxScaler


##############################################################
# SafeSCAD specific functions
# 5 class using mean & sdev
def create_tot_categories(rt_column=None):
    bins = [0, 0, 0, 0, 0, 0]
    labels = np.array(['fast', 'med_fast', 'med', 'med_slow', 'slow'], dtype=object)

    if rt_column is None:
        return (bins,labels)

    rt_mean = round(rt_column.mean())
    rt_sdev = round(rt_column.std())
    bound_1 = rt_mean - rt_sdev
    bound_2 = rt_mean - rt_sdev // 2
    bound_3 = rt_mean + rt_sdev // 2
    bound_4 = rt_mean + rt_sdev
    bins = [float('-inf'), bound_1, bound_2, bound_3, bound_4, float('inf')]
    return (bins, labels)

def upsample_minority_TOTs(X_train, y_train, tot_labels, random_state=27):
    # contat the training data together.
    X = pd.concat([X_train, y_train], axis=1)
    # separate majority and minority classes
    buckets = {l: X[X.TOT == l] for l in tot_labels}
    maj_label, majority = reduce(lambda a,b: b if b[1].shape[0] > a[1].shape[0] else a, buckets.items())
    minorities = {k:v for k,v in buckets.items() if k != maj_label}
    # upsample the minority classes
    for k,v in minorities.items():
        buckets[k] = resample(v, replace=True, n_samples=majority.shape[0], random_state=random_state)
    upsampled = pd.concat(buckets.values()).sample(frac=1)
    # split the upsampled data into X and y
    y_train = upsampled['TOT']
    X_train = upsampled.drop('TOT', axis=1)
    return X_train, y_train

def get_safescad_data(dataset_file, data_dir, noise, splits):
    print('Getting SafeScad dataset...')
    n_categories = len(create_tot_categories()[1])
    if dataset_file is not None:
        " Import Dataset """

        raw_columns = ['ID', 'FixationDuration', 'FixationStart', 'FixationSeq',
                       'FixationX', 'FixationY', 'GazeDirectionLeftZ', 'GazeDirectionRightZ',
                       'PupilLeft', 'PupilRight', 'InterpolatedGazeX', 'InterpolatedGazeY',
                       'AutoThrottle', 'AutoWheel', 'CurrentThrottle', 'CurrentWheel',
                       'Distance3D', 'MPH', 'ManualBrake', 'ManualThrottle', 'ManualWheel',
                       'RangeW', 'RightLaneDist', 'RightLaneType', 'LeftLaneDist', 'LeftLaneType',
                       'ReactionTime']
        raw_df = pd.read_csv(dataset_file, usecols=raw_columns)
        raw_df.set_index(['ID'], inplace=True)

        # make a copy the raw data
        df = raw_df

        # compute 'TOT' categories
        tot_bins, tot_labels = create_tot_categories(df.ReactionTime)
        # print the TOT categories
        print('TOT Categories')
        print('\n'.join(
            ['%s: %9.2f, %7.2f' % (tot_labels[i].rjust(8), tot_bins[i], tot_bins[i + 1]) for i in
             range(n_categories)]))

        # chunk_users = ['015_M3', '015_m2', '015_M1', '014_M3', '014_M2', '014_m1']
        # df = df.loc[df['Name'].isin(chunk_users)]

        df.RightLaneType = df.RightLaneType.astype(int)
        df.LeftLaneType = df.LeftLaneType.astype(int)

        # add the class to the dataframe
        df['TOT'] = pd.cut(df.ReactionTime, bins=tot_bins, labels=tot_labels).astype(object)

        # keep just 10% of the data
        df = df.sample(frac=0.2)

        # split features and targets
        y = df.TOT
        X = df.drop(['ReactionTime', 'TOT'], axis=1)

        # display the feature names
        feature_names = list(X.columns)
        print('Feature Names', feature_names)

        # make results easier to reproduce
        random_state = 27

        # prepare encoders
        scaler = prepare_inputs(X)
        onehot = prepare_target(y, categories=[tot_labels])

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y,
                                                            random_state=random_state)

        # upsample the training data
        print('TOT Value Counts before upsampling\n', y_train.value_counts())
        X_train, y_train = upsample_minority_TOTs(X_train, y_train, tot_labels)
        print('TOT Value Counts after upsampling\n', y_train.value_counts())

        # scale the inputs
        X_train_enc = scaler.transform(X_train.values)
        X_test_enc = scaler.transform(X_test.values)

        # add noise to inputs
        if noise is not None:
            X_train_enc = np.apply_along_axis(add_noise(noise), 1, X_train_enc)

        # categorize outputs
        y_train_enc = onehot.transform(y_train.to_numpy().reshape(-1,1)).toarray()
        y_test_enc = onehot.transform(y_test.to_numpy().reshape(-1,1)).toarray()

        print("Saving train and test data ...")
        save_data(X_train_enc, y_train_enc, feature_names, onehot, 'TOT', 'train', data_dir)
        save_data(X_test_enc, y_test_enc, feature_names, onehot, 'TOT', 'test', data_dir)
        save_encoders(scaler, onehot, data_dir)

        return (X_train_enc, y_train_enc, X_test_enc, y_test_enc)

    else:
        train_data = pd.read_csv(f'{data_dir}/train.csv')
        test_data = pd.read_csv(f'{data_dir}/test.csv')

        y_train_enc = train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        y_test_enc = test_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]

        X_train_enc = train_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        X_test_enc = test_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow','TOT_slow'],axis=1)


        feature_names = list(X_train_enc.columns)
        # display the feature names
        print('Feature Names', feature_names)

        return (X_train_enc.values, y_train_enc.values, X_test_enc.values, y_test_enc.values)


##############################################################
# SafeSCAD2 specific functions

def scale_data(X, output_dir):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    filename = f'{output_dir}/scaler_path'
    pickle.dump(scaler, open(filename, 'wb'))
    return X

def get_scaled_data(df,cols, target_col, output_dir):
    train = pd.DataFrame(df, columns=cols)
    train = train.astype('float32')

    # scale data
    scale_cols = cols.copy()
    scale_cols.remove(target_col)

    train_scaled = scale_data(train[scale_cols], output_dir)
    train_scaled.columns = scale_cols

    X_train = train_scaled.values
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = df.loc[:, [target_col]]
    y_train = enc.fit_transform(y_train, y=None).toarray()
    filename = f'{output_dir}/OHE_path'
    pickle.dump(enc, open(filename, 'wb'))
    return X_train, y_train, enc, train_scaled.columns

def upsample_minority(df, tot_labels, target_col, random_state=27):
    X_train = df.drop(columns = target_col)
    y_train = df[target_col]
    # contat the training data together.
    X = pd.concat([pd.DataFrame(X_train), y_train], axis=1)
    # separate majority and minority classes
    buckets = {l: X[X[target_col] == l] for l in tot_labels}
    maj_label, majority = reduce(lambda a,b: b if b[1].shape[0] > a[1].shape[0] else a, buckets.items())
    minorities = {k:v for k,v in buckets.items() if k != maj_label}
    # upsample the minority classes
    for k,v in minorities.items():
        buckets[k] = resample(v, replace=True, n_samples=majority.shape[0], random_state=random_state)
    upsampled = pd.concat(buckets.values()).sample(frac=1)

    return upsampled

def get_safescad2_data(dataset_file, data_dir, noise, splits):
    print('Getting SafeScad2 dataset...')
    if dataset_file is not None:
        " Import Dataset """

        raw_columns = ['number_of_eye_movements','number_of_look_changes',
                         'number_of_speed_changes','number_of_steer_positions',
                         'percentage_of_screen_look','sum_distance_of_eyes',
                         'sum_distance_of_looks','sum_distance_of_steer',
                         'average_speed','average_GSR_Resistance','average_GSR_Conductance',
                         'average_CurrentBrake','difference_of_average_speed_from_previous_data',
                         'difference_of_average_GSR_Resistance_from_previous_data',
                         'difference_of_average_GSR_Conductance_from_previous_data',
                         'difference_of_average_CurrentBrake_from_previous_data','reaction_time']

        # load the data
        df = pd.read_csv(dataset_file)
        # upscale minority
        tot_labels = [0, 1, 2]  #fast, med, slow
        df = upsample_minority(df, tot_labels, 'reaction_time', random_state=27)
        # clean and scale the data
        X, y, onehot, feature_names = get_scaled_data(df, raw_columns, 'reaction_time', data_dir)

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y,
                                                            random_state=42)

        # add noise to inputs
        if noise is not None:
            X_train = np.apply_along_axis(add_noise(noise), 1, X_train)

        print("Saving train and test data ...")
        save_data(X_train, y_train, feature_names, onehot, 'TOT', 'train', data_dir)
        save_data(X_test, y_test, feature_names, onehot, 'TOT', 'test', data_dir)

        return (X_train, y_train, X_test, y_test)

    else:
        train_data = pd.read_csv(f'{data_dir}/train.csv')
        test_data = pd.read_csv(f'{data_dir}/test.csv')

        y_train_enc = train_data[['TOT_0', 'TOT_1', 'TOT_2']]
        y_test_enc = test_data[['TOT_0', 'TOT_1', 'TOT_2']]

        X_train_enc = train_data.drop(['Unnamed: 0', 'TOT_0','TOT_1', 'TOT_2'], axis=1)
        X_test_enc = test_data.drop(['Unnamed: 0', 'TOT_0', 'TOT_1', 'TOT_2'],axis=1)


        feature_names = list(X_train_enc.columns)
        # display the feature names
        print('Feature Names', feature_names)

        return (X_train_enc.values, y_train_enc.values, X_test_enc.values, y_test_enc.values)
##############################################################

##############################################################
# Generic functions
def prepare_inputs(X):
    # scales inputs using "standard scaler", and returns 2D numpy array
    scaler = StandardScaler().fit(X)
    return scaler

def prepare_target(y, categories):
    # convert target to categorical, and returns 2D numpy array
    y = y.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(categories=categories)
    onehot.fit(y)
    return onehot

def save_encoders(scaler, onehot, output_dir):
    if scaler is not None:
        pkl.dump(scaler, open(f'{output_dir}/scaler.pkl', 'wb'))
    if onehot is not None:
        pkl.dump(onehot, open(f'{output_dir}/onehot.pkl', 'wb'))

def save_data(X_enc, y_enc, features, onehot, onehot_name, fname, data_dir='../experiments/data'):
    labels = onehot.get_feature_names(input_features=[onehot_name])
    df = pd.concat([pd.DataFrame(X_enc, columns=features),
                          pd.DataFrame(y_enc, columns=labels)],
                        axis=1).astype({k:int for k in labels})
    data_csv = f'{data_dir}/{fname}.csv'
    df.to_csv(data_csv)
    print(f'wrote data to {data_csv}')

def add_noise(noise):
    f = lambda x: x + tf.random.normal(tf.shape(x), stddev=noise)
    return f


def get_data(experiment, dataset_file, data_dir, noise=None, splits=None):
    print("Loading and preprocessing data ...")
    get_data = globals()[f'get_{experiment}_data']
    return get_data(dataset_file, data_dir, noise, splits)
