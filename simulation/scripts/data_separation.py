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
import string

#from robomimic.utils.file_utils import create_hdf5_filter_key

### Change these parameters if you want to generate a different distribution of data
max_val = 500
splits = [1, .75, .5, .25, 0]


def local_create_hdf5_filter_key(f, demo_keys, key_name):
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype='S')

    return ep_lengths

def separate_trajectories(args):
    path = args.original_folder
    rand_path = args.rand_folder
    files = os.listdir(path)
    rand_files = os.listdir(rand_path)

    output = args.output
    env_args = None

    with h5py.File("/tmp/rand.hdf5", "w") as tempr:
        data_tempr = tempr.create_group('data')
        total_r = 0

        

        for file in rand_files:
            filename, filetype = os.path.splitext(file)
            if filetype != ".hdf5":
                continue

            with h5py.File(rand_path + file, "r") as f:
                a_group_key = list(f.keys())[0]
                
                demons = list(f[a_group_key].keys())
                total_rand = 0

                if env_args == None:
                    env_args = f['data'].attrs['env_args']

                if env_args != f['data'].attrs['env_args']:
                    raise Exception("Your demonstrations aren't coming from the same enviornment")
                
                for demo_id in demons:
                    demo_data = f[a_group_key].__getitem__(demo_id)
                    demo_name = "demo_{}".format(total_r)

                    tempr.copy(demo_data, data_tempr, demo_name)
                    total_r += 1
                    # rand_demons.append(demo_data.values)
        
        if total_r +1 < max_val:
            raise Exception("You need " + str(max_val - total_r - 1) + " more random demonstrations")

        tempr['data'].attrs['env_args'] = env_args

        tempr['data'].attrs['total'] = total_r+1
        
        tempr.close
                

    with h5py.File("/tmp/success.hdf5", "w") as temps, h5py.File("/tmp/fail.hdf5", "w") as tempf:
        data_temps = temps.create_group('data')
        data_tempf = tempf.create_group('data')

        total_s = 0
        total_f = 0

        for file in files:
            filename, filetype = os.path.splitext(file)
            if filetype != ".hdf5":
                continue
            
            with h5py.File(path+file, "r") as f:

                a_group_key = list(f.keys())[0]

                if env_args != f['data'].attrs['env_args']:
                    raise Exception("Your demonstrations aren't coming from the same enviornment")
                
                demons = list(f[a_group_key].keys())
               
                for demo_id in demons:
                    demo_data = f[a_group_key].__getitem__(demo_id)
                    rewards = np.array(demo_data.get('rewards'))

                    if np.count_nonzero(rewards) > 0:
                        demo_name = "demo_{}".format(total_s)
                        tempr.copy(demo_data, data_temps, demo_name)
                        total_s += 1
                    
                    else:
                        demo_name = "demo_{}".format(total_f)
                        tempr.copy(demo_data, data_tempf, demo_name)
                        total_f += 1

        if total_f +1 < max_val:
            raise Exception("You need " + str(max_val - total_f - 1) + " more failed demonstrations")
        
        if total_s +1 < max_val:
            raise Exception("You need " + str(max_val - total_s - 1) + " more successful demonstrations")

        tempf['data'].attrs['env_args'] = env_args
        temps['data'].attrs['env_args'] = env_args

        tempf['data'].attrs['total'] = total_f
        temps['data'].attrs['total'] = total_s
    
    # fails = len(fail_demons)
    # success = len(success_demons)

    # if fails < max_val:
    #     raise Exception("You need " + str(max_val - fails) + " more failed demonstrations")
    
    # if success < max_val:
    #     raise Exception("You need " + str(max_val - success) + " more successful demonstrations")

    
    
    with h5py.File("/tmp/success.hdf5", "r") as temps, h5py.File("/tmp/fail.hdf5", "r") as tempf, h5py.File("/tmp/rand.hdf5", "r") as tempr:
        # rand_demons = []
        # success_demons = []
        # fail_demons = []

        a_group_key_s = list(temps.keys())[0]
        demons_s = list(temps[a_group_key_s].keys())
               
        # for demo_id in demons:
        #     demo_data = temps[a_group_key].__getitem__(demo_id)
        #     success_demons.append(demo_data)
        
        a_group_key_r = list(tempr.keys())[0]
        demons_r = list(tempr[a_group_key_r].keys())
               
        # for demo_id in demons:
        #     demo_data = tempr[a_group_key].__getitem__(demo_id)
        #     rand_demons.append(demo_data)
        
        a_group_key_f = list(tempf.keys())[0]
        demons_f = list(tempf[a_group_key_f].keys())
               
        # for demo_id in demons:
        #     demo_data = tempf[a_group_key].__getitem__(demo_id)
        #     print(demo_data)
        #     fail_demons.append(demo_data)

        

        # rand_demons = np.asarray(rand_demons)
        # success_demons = np.asarray(success_demons)
        # fail_demons = np.asarray(fail_demons)

        for split in splits:
            full = 1 - split
            half = full/2
            
            fail_name = ("s" + str(split) + "f" + str(full) + "r" + str(0)).replace(".", "")
            rand_name = ("s" + str(split) + "f" + str(0) + "r" + str(full)).replace(".", "")
            fail_rand_name = ("s" + str(split) + "f" + str(half) + "r" + str(half)).replace(".", "")

            if fail_name == rand_name:
                rand_name += "_copy"

            full_f = np.random.choice(len(demons_f), size = int(full*max_val), replace = False)
            half_f = np.random.choice(len(demons_f), size = int(half*max_val), replace = False)

            #full_f = fail_demons[full_f]
            #half_f = fail_demons[half_f]

            full_r = np.random.choice(len(demons_r), size = int(full*max_val), replace = False)
            half_r = np.random.choice(len(demons_r), size = int(half*max_val), replace = False)

            # full_r = rand_demons[full_r]
            # half_r = rand_demons[half_r]

            full_s = np.random.choice(len(demons_s), size = int(split*max_val), replace = False)

            # full_s = success_demons[full_s]


            # fail = np.concatenate((full_s, full_f))
            # np.random.shuffle(fail)

            # print(fail.shape)

            # rand = np.concatenate((full_s, full_r))
            # np.random.shuffle(rand)

            # fail_rand = np.concatenate((full_s, half_f, half_r))
            # np.random.shuffle(fail_rand)

            with h5py.File(output + fail_name + ".hdf5", "w") as ff, h5py.File(output + rand_name + ".hdf5", "w") as fr, h5py.File(output + fail_rand_name + ".hdf5", "w") as ffr:
                
                data_ff = ff.create_group('data')
                data_fr = fr.create_group('data')
                data_ffr = ffr.create_group('data')

                total_ff = len(full_s)
                total_fr = len(full_s)
                total_ffr = len(full_s)

                shuffled = np.random.choice(len(demons_r), size = len(demons_r), replace = False)
                shuffled = np.random.choice(max_val, size = max_val, replace = False)

                print(shuffled)
                
                
                for i, x in enumerate(full_s):
                    demo_id = demons_s[x]
                    demo_data = temps[a_group_key_s].__getitem__(demo_id)
                    demo_name = "demo_{}".format(shuffled[i])

                    ff.copy(demo_data, data_ff, demo_name)
                    fr.copy(demo_data, data_fr, demo_name)
                    ffr.copy(demo_data, data_ffr, demo_name)

                for i, x in enumerate(full_f):
                    demo_id = demons_f[x]
                    demo_data = tempf[a_group_key_f].__getitem__(demo_id)
                    demo_name = "demo_{}".format(shuffled[total_ff])

                    total_ff += 1
                    ff.copy(demo_data, data_ff, demo_name)
                
                for i, x in enumerate(full_r):
                    demo_id = demons_r[x]
                    demo_data = tempr[a_group_key_r].__getitem__(demo_id)
                    demo_name = "demo_{}".format(shuffled[total_fr])

                    total_fr += 1
                    fr.copy(demo_data, data_fr, demo_name)
                
                for i, x in enumerate(half_f):
                    demo_id = demons_f[x]
                    demo_data = tempf[a_group_key_f].__getitem__(demo_id)
                    demo_name = "demo_{}".format(shuffled[total_ffr])

                    total_ffr += 1
                    ffr.copy(demo_data, data_ffr, demo_name)
                
                for i, x in enumerate(half_r):
                    demo_id = demons_r[x]
                    demo_data = tempr[a_group_key_r].__getitem__(demo_id)
                    demo_name = "demo_{}".format(shuffled[total_ffr])

                    total_ffr += 1
                    ffr.copy(demo_data, data_ffr, demo_name)




                # for demo_id in shuffled:
                # #for demo_id in range(len(demons_f)):


                #     demo_name = "demo_{}".format(demo_id)
                
                #     ff.copy(fail[demo_id], data_ff, demo_name)
                #     fr.copy(rand[demo_id], data_fr, demo_name)
                #     ffr.copy(fail_rand[demo_id], data_ffr, demo_name)

                ff['data'].attrs['env_args'] = env_args
                fr['data'].attrs['env_args'] = env_args
                ffr['data'].attrs['env_args'] = env_args



                ff['data'].attrs['total'] = max_val
                fr['data'].attrs['total'] = max_val
                ffr['data'].attrs['total'] = max_val
            

                #Adding Filter Keys
                num_demos = max_val
                train = int(num_demos * .9)
                valid = num_demos - train

                percent_20 = int(num_demos * .2)
                percentTrain_20 = int(train * .2)
                percentValid_20 = int(valid * .2)

                percent_50 = int(num_demos * .5)
                percentTrain_50 = int(train * .5)
                percentValid_50 = int(valid * .5)

                demons = []
                for i in range(max_val):
                    demons.append("demo_{}".format(i))
                print(demons)
                for x in [ff, fr, ffr]:

                    local_create_hdf5_filter_key(x, demons[:train] , "train")
                    local_create_hdf5_filter_key(x, demons[train:] , "valid")

                    local_create_hdf5_filter_key(x, demons[:percent_20] , "20_percent")
                    local_create_hdf5_filter_key(x, demons[:percent_50] , "50_percent")

                    local_create_hdf5_filter_key(x, demons[:percentTrain_20] , "20_percent_train")
                    local_create_hdf5_filter_key(x, demons[:percentTrain_50] , "50_percent_train")

                    local_create_hdf5_filter_key(x, demons[train:train+percentValid_20] , "20_percent_valid")
                    local_create_hdf5_filter_key(x, demons[train:train+percentValid_50] , "50_percent_valid")
                
                ff.close
                fr.close
                ffr.close
        temps.close
        tempf.close
        tempr.close

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_folder",
        type=str,
        help="path of the folder containing all the hdf5 files needed to be separated",
    ) 
    parser.add_argument(
        "--rand_folder",
        type=str,
        help="path of the folder containing all the hdf5 files needed to be separated",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path for outputing all hdf5 files",
    )

    args = parser.parse_args()
    separate_trajectories(args)

