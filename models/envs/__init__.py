from gym.envs.registration import register

register(
    id="PandaSwing-v0",
    entry_point="models.envs.panda_swing_env:PandaSwingEnv",
)