from robomimic.utils.file_utils import create_hdf5_filter_key
import h5py

path = "RLS_project/small_traj/H150.hdf5"
with h5py.File(path, "r") as f:
    a_group_key = list(f.keys())[0]
    demons = list(f[a_group_key].keys())

num_demos = len(demons)
train = int(num_demos * .9)
valid = num_demos - train

percent_20 = int(num_demos * .2)
percentTrain_20 = int(train * .2)
percentValid_20 = int(valid * .2)

percent_50 = int(num_demos * .5)
percentTrain_50 = int(train * .5)
percentValid_50 = int(valid * .5)


create_hdf5_filter_key(path, demons[:train] , "train")
create_hdf5_filter_key(path, demons[train:] , "valid")

create_hdf5_filter_key(path, demons[:percent_20] , "20_percent")
create_hdf5_filter_key(path, demons[:percent_50] , "50_percent")

create_hdf5_filter_key(path, demons[:percentTrain_20] , "20_percent_train")
create_hdf5_filter_key(path, demons[:percentTrain_50] , "50_percent_train")

create_hdf5_filter_key(path, demons[train:train+percentValid_20] , "20_percent_valid")
create_hdf5_filter_key(path, demons[train:train+percentValid_50] , "50_percent_valid")