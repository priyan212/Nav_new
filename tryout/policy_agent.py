"""NavDP_Agent adapter -- makes tube_planner/ + navdp_s2diff_server.py runnable
against THIS repo's actual NavDP model, since the officially released NavDP
baseline's own policy_agent.py (and its matching cross-waic-*.ckpt checkpoint)
were never part of this repo.

tube_planner/s2diff_agent.py and tube_planner/s2diff_guidance.py are used
unmodified. They were written against the released NavDP_Agent's interface;
this class reproduces just enough of that interface -- reverse-engineered
from how s2diff_agent.py/s2diff_guidance.py actually call `base_agent`/
`policy` -- to drive nav_pipeline/navdp_net.py's NavDPStandalone instead,
loaded from checkpoints/navdp_extracted.pth (the same weights the real rover
pipeline runs).

Known deviations from what the official agent presumably does:
  * memory_size/predict_size passed in by navdp_s2diff_server.py's argparse
    defaults (8/24, tuned for the released checkpoint) are IGNORED and
    forced to (2/32) -- the shapes navdp_extracted.pth was actually trained
    with. Passing the server's defaults through unchanged would silently
    load garbage into (or crash on) the positional-embedding layers, which
    are shape-locked to memory_size/predict_size at construction time.
  * predict_noise() only supports batch=1 (a single robot/environment).
    NavDPStandalone._predict_noise() internally re-repeats a batch=1
    conditioning tensor to match the action batch (see nav_pipeline/
    navdp_net.py), unlike a natively batch-general implementation, so this
    adapter takes the first row of whatever pre-repeated goal/rgbd embedding
    it's given (repeat_interleave duplicates identical rows, so for batch=1
    this reproduces the same result) and raises if more than one real
    environment is passed.
  * goal_point sign convention: navdp_extracted.pth's point_encoder was
    trained on the checkpoint's native [x fwd, y RIGHT, z] convention (the
    same quirk nav_pipeline/pipeline.py works around), while
    navdp_s2diff_server.py's goal_x/goal_y JSON fields are [x fwd, y LEFT]
    (ROS-style, matching S2DIFF_GUIDANCE.md's obstacle-pixel frame) -- so
    process_pointgoal() flips y before encoding.
  * project_trajectory() is a stub. navdp_s2diff_server.py's /pointgoal_step
    route discards step_pointgoal()'s 4th return value entirely, so nothing
    ever needs a real visualization mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nav_pipeline.navdp_net import NavDPStandalone  # noqa: E402

_TRAINED_MEMORY_SIZE = 2
_TRAINED_PREDICT_SIZE = 32


class _NaviFormerView:
    """Adapts NavDPStandalone to the numpy-in / pre-repeated-conditioning
    interface tube_planner/s2diff_guidance.py's sample_*_pointgoal_candidates
    call sites expect from `policy`."""

    def __init__(self, model: NavDPStandalone):
        self._model = model
        self.predict_size = model.predict_size
        self.noise_scheduler = model.noise_scheduler
        self.device = str(model._device)

    def _to_model_tensor(self, x):
        # x may already be a (possibly CUDA) tensor (this adapter's own
        # process_depth) or a plain numpy array (tube_planner/s2diff_agent.py's
        # np.asarray(input_images)) -- np.asarray() can't touch a CUDA
        # tensor, so branch instead of assuming either.
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self._model.input_dtype, device=self._model._device)
        return torch.as_tensor(x, dtype=self._model.input_dtype, device=self._model._device)

    def rgbd_encoder(self, images, depths):
        return self._model.rgbd_encoder(self._to_model_tensor(images), self._to_model_tensor(depths))

    def point_encoder(self, goals):
        goals_t = torch.as_tensor(goals, dtype=self._model.input_dtype, device=self._model._device)
        return self._model.point_encoder(goals_t)

    def predict_noise(self, actions, timestep, goal_embed, rgbd_embed):
        if goal_embed.shape[0] != 1 or rgbd_embed.shape[0] != 1:
            # tube_planner/s2diff_guidance.py pre-repeats identical rows via
            # torch.repeat_interleave for batch>1; NavDPStandalone only
            # supports batch=1 (see module docstring), so recover the single
            # unique row rather than silently mixing multiple environments.
            if not (torch.allclose(goal_embed, goal_embed[:1].expand_as(goal_embed))
                    and torch.allclose(rgbd_embed, rgbd_embed[:1].expand_as(rgbd_embed))):
                raise ValueError("this NavDP_Agent adapter only supports batch_size=1")
            goal_embed = goal_embed[:1]
            rgbd_embed = rgbd_embed[:1]
        # tube_planner/s2diff_guidance.py runs its DDPM loop in float32;
        # NavDPStandalone's layers are bf16 -- cast in, cast back out so the
        # rest of that float32-native loop can keep operating on the result.
        input_dtype = actions.dtype
        actions_t = actions.to(dtype=self._model.input_dtype)
        noise = self._model._predict_noise(actions_t, timestep, goal_embed, rgbd_embed)
        return noise.to(dtype=input_dtype)

    @property
    def critic_head(self):
        return self._model.critic_head

    @critic_head.deleter
    def critic_head(self):
        del self._model.critic_head


class NavDP_Agent:
    def __init__(
        self,
        intrinsic,
        image_size: int = 224,
        memory_size: int = 8,
        predict_size: int = 24,
        temporal_depth: int = 16,
        heads: int = 8,
        token_dim: int = 384,
        navi_model: str = "",
        device: str = "cuda:0",
    ):
        if memory_size != _TRAINED_MEMORY_SIZE or predict_size != _TRAINED_PREDICT_SIZE:
            print(
                f"[policy_agent] navdp_extracted.pth was trained with "
                f"memory_size={_TRAINED_MEMORY_SIZE}, predict_size={_TRAINED_PREDICT_SIZE} -- "
                f"overriding the requested ({memory_size}, {predict_size})"
            )
        self.image_size = image_size
        self.memory_size = _TRAINED_MEMORY_SIZE
        self.predict_size = _TRAINED_PREDICT_SIZE
        self.device = device
        self.image_intrinsic = np.asarray(intrinsic, dtype=np.float32)

        self.model = NavDPStandalone.load(
            weights_path=navi_model,
            device=device,
            image_size=image_size,
            memory_size=self.memory_size,
            predict_size=self.predict_size,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
        )
        self.navi_former = _NaviFormerView(self.model)

        self.batch_size = 0
        self.stop_threshold = 0.0
        self.memory_queue: list[list[np.ndarray]] = []
        self._depth_queue: list[list[np.ndarray]] = []

    # ------------------------------------------------------------------ #
    def reset(self, batch_size: int, stop_threshold: float) -> None:
        self.batch_size = batch_size
        self.stop_threshold = stop_threshold
        self.memory_queue = [[] for _ in range(batch_size)]
        self._depth_queue = [[] for _ in range(batch_size)]

    def reset_env(self, index: int) -> None:
        self.memory_queue[index] = []
        self._depth_queue[index] = []

    # ------------------------------------------------------------------ #
    def process_image(self, images: np.ndarray) -> np.ndarray:
        """images: (B,H,W,3) BGR uint8 (navdp_s2diff_server.py's own
        convention, see _decode_request) -> (B,image_size,image_size,3)
        float32 RGB/255, matching NavDPStandalone's input convention."""

        out = np.empty((images.shape[0], self.image_size, self.image_size, 3), dtype=np.float32)
        for i in range(images.shape[0]):
            rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            out[i] = cv2.resize(rgb, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        return out

    def process_depth(self, depths: np.ndarray) -> torch.Tensor:
        """depths: (B,H,W,1) metric meters -> (B,memory_size,image_size,
        image_size,1) torch tensor, maintaining a per-env depth history the
        same way s2diff_agent.py maintains its own image memory_queue (NavDP
        StandAlone's rgbd_encoder needs images and depths at the same
        memory_size)."""

        current = np.empty((depths.shape[0], self.image_size, self.image_size, 1), dtype=np.float32)
        for i in range(depths.shape[0]):
            d = cv2.resize(depths[i, ..., 0], (self.image_size, self.image_size))
            d = np.where(np.isfinite(d) & (d >= 0.1) & (d <= 5.0), d, 0.0).astype(np.float32)
            current[i] = d[..., None]

        stacked = []
        for i, queue in enumerate(self._depth_queue):
            if len(queue) < self.memory_size:
                queue.append(current[i])
                hist = np.stack(queue, axis=0)
                pad = self.memory_size - hist.shape[0]
                if pad:
                    hist = np.pad(hist, ((pad, 0), (0, 0), (0, 0), (0, 0)))
            else:
                del queue[0]
                queue.append(current[i])
                hist = np.stack(queue, axis=0)
            stacked.append(hist)
        return torch.as_tensor(np.stack(stacked, axis=0), dtype=self.model.input_dtype, device=self.model._device)

    def process_pointgoal(self, goals: np.ndarray) -> np.ndarray:
        """goals: (B,3) [x fwd, y LEFT, z] -> checkpoint-native [x fwd, y
        RIGHT, z] (see module docstring's sign-convention note). Returned as
        plain numpy -- sample_s2diff_pointgoal_candidates does its own
        np.asarray(...)/torch.as_tensor(..., device=...) on this value."""

        flipped = np.asarray(goals, dtype=np.float32).copy()
        flipped[:, 1] = -flipped[:, 1]
        return flipped

    def project_trajectory(self, images, all_trajectories, all_values):
        """Stub: navdp_s2diff_server.py's /pointgoal_step discards this
        return value entirely, so no real visualization is computed."""

        return None
