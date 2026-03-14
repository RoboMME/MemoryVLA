"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from robomme.robomme_env import *  # noqa: F401, F403 - env registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from utils import TASK_NAME_LIST

np.set_printoptions(precision=8, suppress=True)


def pack_state(eef: np.ndarray, grip: np.ndarray) -> np.ndarray:
    return np.concatenate([eef, grip], axis=0, dtype=np.float32)


class EnvRunner:
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API, and optional subgoal oracles.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1300) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        self.video_save_dir = video_save_dir
        self.action_space = "ee_pose"
        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space=self.action_space, # MemoryVLA uses ee_pose action space
            gui_render=False,
            max_steps=max_steps,
        )

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.difficulty: Any = None
        self.task_goal: str = ""

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id

    def get_init_obs(self) -> dict[str, Any]:
        """Reset env and return initial observation dict (images, wrist_images, states, task_goal)."""
        self.obs, self.info = self.env.reset()        
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]

        images =  self.obs["front_rgb_list"]
        wrist_images = self.obs["wrist_rgb_list"]
        states = [pack_state(eef, grip) for eef, grip in zip(self.obs["eef_state_list"], self.obs["gripper_state_list"])]

        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_goal": self.task_goal,
        }

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        """
        Execute one step.
        Returns ( (img, wrist_img, state), stop_flag, success_flag ).
        success_flag is one of "success", "fail", "timeout", "error".
        """
        
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        if self.obs is None:
            return (None, None, None), True, "error"
        
        img = self.obs["front_rgb_list"][-1]
        wrist_img = self.obs["wrist_rgb_list"][-1]
        state = pack_state(self.obs["eef_state_list"][-1], self.obs["gripper_state_list"][-1])
        outcome = self.info.get("status", "error")
        stop = (terminated or truncated)
                
        return (img, wrist_img, state), stop, outcome
    
    def get_current_state(self):
        if self.action_space == "joint_angle":
            return np.array(self.obs["joint_state_list"][-1])
        elif self.action_space == "ee_pose":
            return np.array(self.obs["eef_state_list"][-1])
        else:
            raise ValueError(f"Invalid action space: {self.action_space}")

    def close_env(self) -> None:
        """Close and clear the current env."""
        if self.env is not None:
            self.env.close()
            del self.env
            self.env = None