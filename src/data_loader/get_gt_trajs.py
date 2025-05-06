#%%

import argparse

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (
    Origin, Space, StatefulTractogram)
from dipy.io.vtk import load_vtk_streamlines
import numpy as np
import nibabel as nib
import os
import sys
sys.path.insert(0, '/tracto/TrackToLearn')
sys.path.insert(0, '/tracto')

# Try direct imports from the files
from environments.env import BaseEnv
from environments.reward import reward_streamlines_step
import pickle

def main(bundle: str):


    subs_list = ['sub-1160']
    print('Generating trajectories')
    # %%

    def get_tractogram_in_voxel_space(
        tract_fname,
        ref_anat_fname,
        tracts_attribs={'orientation': 'unknown'},
        origin=Origin.NIFTI
    ):
        if tracts_attribs['orientation'] != 'unknown':
            to_lps = tracts_attribs['orientation'] == 'LPS'
            streamlines = load_vtk_streamlines(tract_fname, to_lps)
            sft = StatefulTractogram(streamlines, ref_anat_fname, Space.RASMM)
        else:
            sft = load_tractogram(
                tract_fname, ref_anat_fname,
                bbox_valid_check=False, trk_header_check=False)
        sft.to_vox()
        sft.to_origin(origin)
        return sft


    ttoi_path = '/tracto/TractoDiff/data/'

    def get_subject_ids(folder, split):
        subjects_path = os.path.join(folder, split)
        if not os.path.exists(subjects_path):
            raise FileNotFoundError(f"The path '{subjects_path}' does not exist. Only split 'trainset' or 'testset' or 'validset' are allowed.")
        subjects_list = os.listdir(subjects_path)
        return subjects_list


    splits = ['testset']
    all_subs_dict = {split: get_subject_ids(ttoi_path, split) for split in splits}

    def find_key_by_element(dictionary, target_element):
        for key, value_list in dictionary.items():
            if target_element in value_list:
                return key
        return None

    # %%

    for sub_id in subs_list:
        split = find_key_by_element(all_subs_dict, sub_id)
        tract_fname = f'/tracto/TractoDiff/data/{split}/{sub_id}/tractography/{sub_id}__{bundle}.trk'
        ref_anat_fname = f'/tracto/TractoDiff/data/{split}/{sub_id}/dti/{sub_id}__fa.nii.gz'
        tractogram_voxel_space = get_tractogram_in_voxel_space(tract_fname, ref_anat_fname)

        # space check:-
        og_sft = load_tractogram(
                    tract_fname, ref_anat_fname,
                    bbox_valid_check=False, trk_header_check=False)
        print(f'original trk space: {og_sft.space}')
        print(f'modified trk space: {tractogram_voxel_space.space}')

        dataset_file = f'/tracto/TractoDiff/data/{split}/{sub_id}/{sub_id}.hdf5'
        wm_loc = f'/tracto/TractoDiff/data/{split}/{sub_id}/{sub_id}-generated_approximated_mask.nii.gz'
        target = nib.load(wm_loc).get_fdata() #target and exclude are anyway not used in reward calculation
        exclude = nib.load(f'/tracto/TractoDiff/data/{split}/{sub_id}/mask/{sub_id}__mask_csf.nii.gz').get_fdata()

        env = BaseEnv(dataset_file,
                wm_loc,
                sub_id,
                n_signal= 1,
                n_dirs = 8,
                step_size = 0.2,
                max_angle = 60,
                min_length = 10,
                max_length = 200,
                n_seeds_per_voxel = 4,
                rng= np.random.RandomState(seed=1337),#default of TtoL
                add_neighborhood = 1.5,
                compute_reward = True,
                device= None
            )

        # %%
        streamlines = tractogram_voxel_space.streamlines
        print(f"{sub_id}'s num trajs for {bundle} are: {len(streamlines)}")
        step_size = 0.375 #can change to 0.5 and see ----

        # Formatting in pkl structure of trajs:-

        dicts_lst = []

        for strl in streamlines:
            traj_dict = {'observations': None, 'length': None}
            strl_state_list = []
            
            # Store the length (number of points) in the streamline
            strl_length = strl.shape[0]  # This gives us the number of points
            traj_dict['length'] = strl_length
            
            for l in range(1, strl.shape[0]+1):
                seg = strl[:l].reshape((1, l, 3))
                st = env._format_state(seg)
                st = st.reshape(346,)
                strl_state_list.append(st)
                
            # states:-
            traj_obs = np.array(strl_state_list)
            traj_dict['observations'] = traj_obs[:-1, :]
            del strl_state_list
            
            dicts_lst.append(traj_dict)

        del traj_dict
        #%%

        # Saving pkl:-
        if not os.path.exists(f'/tracto/TractoDiff/output/{bundle}'):
            os.makedirs(f'/tracto/TractoDiff/output/{bundle}')
        save_loc = f'/tracto/TractoDiff/output/{bundle}/{sub_id}.pkl'
        
        with open(save_loc, 'wb') as file:
            print("observations shape:", dicts_lst[0]['observations'].shape)
            print("length of first streamline:", dicts_lst[0]['length'])
            pickle.dump(dicts_lst, file)

        print(f"Data has been saved to {save_loc}")

        del dicts_lst


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the script with a string and a boolean argument.")
    
    # Add a string argument
    parser.add_argument("bundle", type=str, help="The bundle argument.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args.bundle)


