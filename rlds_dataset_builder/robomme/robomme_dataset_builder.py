from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from robomme.conversion_utils import MultiThreadedDatasetBuilder
np.set_printoptions(precision=4, suppress=True)


def calculate_delta_action(action, state):
    # this is EEF action minus EEF state
    # since we use euler angles for orientation, we need to convert them carefully rather than a simple minus
    
    pos_delta = action[:3] - state[:3]
    ori_delta_raw = action[3:6] - state[3:6]
    # Wrap orientation differences to [-π, π] to handle angle periodicity
    # This prevents large differences when angles wrap around (e.g., π and -π)
    ori_delta = np.arctan2(np.sin(ori_delta_raw), np.cos(ori_delta_raw))
    
    # if np.linalg.norm(ori_delta - ori_delta_raw) > 0.01:
    #     print(f"Orientation difference wrapped: {ori_delta_raw} -> {ori_delta}")
    #     print(f'{state[3:6]} -> {action[3:6]}')
    
    gripper_action = action[6]
    return np.concatenate([pos_delta, ori_delta, [gripper_action]], axis=0)

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(data_path, episode_idx):        
        data = h5py.File(data_path, "r")
        episode_dataset = data[f"episode_{episode_idx}"]
        task_goal = episode_dataset["setup"]["task_goal"][()][0].decode().lower()
        
        timestep_indexs = sorted(
            int(k.split("_")[-1])
            for k in episode_dataset.keys()
            if k.startswith("timestep_")
        )
            
        episode = []
        for idx in timestep_indexs:
            ts = episode_dataset[f"timestep_{idx}"]
            gripper_state = ts["obs"]["gripper_state"][()]
            EEF_state = ts["obs"]["eef_state"][()]
            image = ts["obs"]["front_rgb"][()]
            wrist_image = ts["obs"]["wrist_rgb"][()]
            action = ts["action"]["eef_action"][()]
            is_video_demo = ts["info"]["is_video_demo"][()] 
            delta_action = calculate_delta_action(action, EEF_state)
    
            assert delta_action.shape == (7,)
            assert EEF_state.shape == (6,)
            assert gripper_state.shape == (2,)
            assert image.shape == (256, 256, 3)
            assert wrist_image.shape == (256, 256, 3)
                                
            episode.append({
                'observation': {
                    'image': image.astype(np.uint8),
                    'wrist_image': wrist_image.astype(np.uint8),
                    'EEF_state': EEF_state.astype(np.float32),
                    'gripper_state': gripper_state.astype(np.float32),
                },
                'action': delta_action.astype(np.float32),
                'discount': 1.0,
                'reward': float(idx == (timestep_indexs[-1])),
                'is_first': idx == timestep_indexs[0],
                'is_last': idx == timestep_indexs[-1],
                'is_terminal': idx == timestep_indexs[-1],
                'language_instruction': task_goal,
                'is_video_demo': is_video_demo.astype(bool),
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': data_path+f"_{episode_idx}",
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        print(f"Saved episode {episode_idx} of {path}")
        return data_path+f"_{episode_idx}", sample


    for path in paths:
        env_id = path.split("/")[-1].split(".")[0].split("_")[-1]
        print(env_id)
        for episode_idx in range(100):
            yield _parse_example(path, episode_idx)


class ROBOMME(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'EEF_state': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose).',
                        ),
                        'gripper_state': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='Robot gripper state (2D gripper).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'is_video_demo': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if the demo is a video demo, False if it is a teleop demo.'
                    ), 
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob('/data/daiyp/robomme_data_h5_toy/*.h5'),
        }
