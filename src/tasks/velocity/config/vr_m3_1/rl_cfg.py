# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL configuration for VR H3.1 12DOF velocity task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def vr_m3_1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VR M3 1 velocity task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="vr_m3_1_velocity",
        logger="wandb",  # or "tensorboard"
        wandb_project="vr_m3_1_velocity",
        save_interval=1000,
        num_steps_per_env=32,
        max_iterations=20_001,
    )


def vr_m3_1_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """RL config cho task đứng vững (giai đoạn 0, ngân sách ~10 phút).

    Giống hệt bản velocity trừ ba điểm hợp với run rất ngắn:
      - ``save_interval=25``: run chỉ vài trăm iteration, để 1000 như bản gốc
        thì gần như không có checkpoint nào được ghi giữa chừng.
      - ``max_iterations=300``: chỉ là mặc định an toàn; giá trị thật do
        ``scripts/phase0_measure.py`` tính từ ngân sách phút rồi ghi đè.
      - log/W&B tách riêng để không lẫn với run velocity.
    """
    cfg = vr_m3_1_ppo_runner_cfg()
    cfg.experiment_name = "vr_m3_1_stand"
    cfg.wandb_project = "vr_m3_1_stand"
    cfg.save_interval = 25
    cfg.max_iterations = 300
    return cfg
