from collections import deque
import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple
import copy
import numpy as np
import cv2

from utils import (
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
)
from utils import RolloutRecorder
from env_runner import EnvRunner

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False
    obs_horizon: int = 8   # 8 is more stable than 16 from our experiments

    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma



class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
    
    def get_abs_action(self, delta_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        # delta action is 6+1 dimensions
        # current state is 6 dimensions
        xyz = current_state[:3] + delta_action[:3]
        rpy_state = copy.deepcopy(current_state[3:6])
        rpy_state[0] += np.pi*2
        rpy = rpy_state + delta_action[3:6]
        return np.concatenate([xyz, rpy, [delta_action[6]]], axis=0)
        
        

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        video_save_dir: Path,
    ) -> str:
        from vla_policy import LLaVAClient
        client = LLaVAClient(base_url=f'http://localhost:{self.args.port}')
        client.reset()

        pre_traj, recorder = self.init_episode(env_runner, video_save_dir)
        prompt = pre_traj["task_goal"]
        episode_first_frame = 'True'
        
        # For video-conditioned tasks, we need to rollout MemoryVLA on the video inputs
        # to prepare the memory bank for execution.
        length = len(pre_traj["images"])
        if env_runner.env_id in TASK_WITH_VIDEO_DEMO:
            for i in range(0, length - 1, self.args.obs_horizon):
                observation = {
                    "base_cam": cv2.cvtColor(pre_traj["images"][i], cv2.COLOR_BGR2RGB),
                    "states": pre_traj["states"][i],
                }
                client.process_frame(
                    text=prompt,
                    episode_first_frame=episode_first_frame,
                    **observation)
                episode_first_frame = 'False'
        
        
        img, wrist_img, robot_state = pre_traj["images"][-1], pre_traj["wrist_images"][-1], pre_traj["states"][-1]

        action_plan = deque()
        observation = {
            "base_cam": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "states": robot_state,
        }
        count = 0
        

        while True:
            if not action_plan:                
                delta_actions = client.process_frame(
                    text=prompt,
                    episode_first_frame=episode_first_frame,
                    **observation)
                
                episode_first_frame = 'False'
                delta_action_chunk = self.get_action_chunk(delta_actions, obs_horizon=self.args.obs_horizon)
                action_plan.extend(delta_action_chunk)

            delta_action = action_plan.popleft()
            current_state = env_runner.get_current_state()
            action = self.get_abs_action(delta_action, current_state)
            
            obs, stop_flag, outcome = env_runner.step(action)            
            
            img, wrist_img, robot_state = obs
            if img is not None:
                recorder.record(
                    image=img.copy(),
                    wrist_image=wrist_img.copy(),
                    state=robot_state.copy(),
                    action=action.copy(),
                )
                
                observation = {
                    "base_cam": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    "states": robot_state,
                }
            count += 1

            if stop_flag:
                break

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{outcome}_{prompt}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        return outcome


    def init_episode(
        self,
        env_runner: EnvRunner,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]
        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)
        print(f"\ntask_goal: {task_goal}")
        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
            )

        return pre_traj, recorder

    def get_action_chunk(
        self,
        action: str,
        obs_horizon: int,
    ) -> list:
        if ';' in action:
            action = action.replace(';', ' ')

        action = action.split(' ')
        action = [float(x) for x in action]
        action = np.array(action, dtype=float)
        action_chunk = np.reshape(action, newshape=(-1, 7))
        return action_chunk[:obs_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.policy_name
        / f"ckpt{args.model_ckpt_id}"
        / f"seed{args.model_seed}"
    )

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate(args: Args):
    """Main evaluation function."""

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    evaluator = EpisodeEvaluator(args, save_dir)

    for task_name in task_names:
        if task_name not in log_dict:
            log_dict[task_name] = {}

        env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
        num_episodes = env_runner.num_episodes

        for episode_id in range(num_episodes):
            if str(episode_id) in log_dict[task_name]:
                print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                continue

            # try at most 3 times if IK errors
            for try_num in range(1, 4):
                env_runner.make_env(episode_id)
                print(f"[robomme] env for task {task_name} episode {episode_id} setup finished")
                if try_num > 1:
                    print(f"[robomme] trying again for episode {episode_id} (try {try_num})")
                
                try:
                    outcome = evaluator.eval_each_episode(env_runner, video_save_dir)
                except Exception as e:
                    print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                    outcome = "error"
                
                env_runner.close_env()
                if outcome != "error":
                    break

            log_dict[task_name][episode_id] = outcome == "success"
            with open(save_dir / "progress.json", "w") as f:
                json.dump(log_dict, f, indent=2)

        del env_runner
        time.sleep(1)

    try:
        final_results = {}
        final_results["success_rate"] = {
            task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
            for task_name in log_dict.keys()
        }
        final_results["total_success_rate"] = (
            sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
        )
        with open(save_dir / "log.json", "w") as f:
            json.dump(final_results, f, indent=2)
    except Exception as e:
        print(f"Error saving final results: {e}")
        time.sleep(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
