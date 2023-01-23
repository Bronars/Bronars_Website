"""
A script to separate trajectories in a given hdf5 files into successful demonstrations and failed demonstrations.
Note that this will mess with original filter keys and new ones will be made for the produced hdf5 files

Args:
    original_folder(str): path of the folder storing all the hdf5 files waiting to be separated
    output_successful(str): path of the hdf5 file with all the successful trajectories
    output_failed(str): path of the hdf5 file with all the failed trajectories

Example: we have a folder named "Folder" containing multiple hdf5 files.
         Trajectories in all those hdf5 files need to be separated.
         Successful trajectories will be stored in a file named "f1.hdf5", and failed trajectories will be stored in a file named "f2.hdf5"
         
   python data_separation.py --original_folder "Folder//"  --output_successful "f1.hdf5"  --output_failed "f2.hdf5"
"""

import os
import h5py
import argparse
import numpy as np

def separate_trajectories(args):
    path = args.original_folder
    files = os.listdir(path)

    path_successful = args.output_successful
    path_failed = args.output_failed

    for file in files:
        filename, filetype = os.path.splitext(file)
        if filetype != ".hdf5":
            continue

        with h5py.File(path+file, "r") as f, h5py.File(path_successful+filename+"_success.hdf5", "w") as fs, h5py.File(path_failed+filename+"_failed.hdf5", "w") as ff:

            data_s = fs.create_group('data')
            data_f = ff.create_group('data')

            demo_id_f = 0
            demo_id_s = 0

            #print(f['data'].attrs.keys())
            # if "mask" in f:
            #     print("hello")
            #     f.copy("mask", ff)
            #     f.copy("mask", fs)


            a_group_key = list(f.keys())[0]
            #print(list(f.keys()))
            #print(f['data']['demo_0'].keys())
            demons = list(f[a_group_key].keys())
            total_fs = 0
            total_ff = 0
            for demo_id in demons:
                demo_data = f[a_group_key].__getitem__(demo_id)
                rewards = np.array(demo_data.get('rewards'))
                len_data = len(rewards)
                if np.count_nonzero(rewards) > 0:
                    demo_name = "demo_{}".format(demo_id_s)
                    demo_id_s = demo_id_s + 1
                    fs.copy(demo_data,data_s,demo_name)
                    total_fs += len_data
                else :
                    demo_name = "demo_{}".format(demo_id_f)
                    demo_id_f = demo_id_f + 1
                    ff.copy(demo_data,data_f,demo_name)
                    total_ff += len_data
            ff['data'].attrs['env_args'] = f['data'].attrs['env_args']
            fs['data'].attrs['env_args'] = f['data'].attrs['env_args']
            ff['data'].attrs['total'] = total_ff
            fs['data'].attrs['total'] = total_fs
            print(total_fs)
            print(total_ff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_folder",
        type=str,
        help="path of the folder containing all the hdf5 files needed to be separated",
    )

    parser.add_argument(
        "--output_successful",
        type=str,
        default=None,
        help="path of the output hdf5 file containing successful trajectories",
    )

    parser.add_argument(
        "--output_failed",
        type=str,
        default=None,
        help="path of the output hdf5 file containing failed trajectories",
    )
    args = parser.parse_args()
    separate_trajectories(args)

