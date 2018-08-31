from gym.envs.registration import register

register(
    id='CarRacingPixels-v1',
    entry_point='car_env.car_racing:CarRacing',
    kwargs={'use_internal_state': False},
    max_episode_steps=1000,
    reward_threshold=900,
)

register(
    id='CarRacingPixels-Debug-v1',
    entry_point='car_env.car_racing:CarRacing',
    kwargs={'use_internal_state': False, 'draw_debug': True},
    max_episode_steps=1000,
    reward_threshold=900,
)

register(
    id='CarRacingInternalState-v1',
    entry_point='car_env.car_racing:CarRacing',
    kwargs={'use_internal_state': True},
    max_episode_steps=1000,
    reward_threshold=900,
)

register(
    id='CarRacingInternalState-Debug-v1',
    entry_point='car_env.car_racing:CarRacing',
    kwargs={'use_internal_state': True, 'draw_debug': True},
    max_episode_steps=1000,
    reward_threshold=900,
)
