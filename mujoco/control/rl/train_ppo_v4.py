"""Train the safe-action PPO v4 controller for Panda joint 2."""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from panda_joint2_env_v4 import PandaJoint2EnvV4


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rl" / "ppo_v4"
MODEL_PATH = OUTPUT_DIR / "models" / "ppo_joint2_v4_100k"
MONITOR_PATH = OUTPUT_DIR / "logs" / "train.monitor.csv"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    check_env(PandaJoint2EnvV4(), warn=True)

    train_env = Monitor(
        PandaJoint2EnvV4(),
        filename=str(MONITOR_PATH),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            }
        },
        verbose=1,
        device="cpu",
        seed=42,
    )

    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(str(MODEL_PATH))
    train_env.close()

    print(f"PPO v4 model saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
