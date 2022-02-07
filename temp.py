# import gym
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# env = gym.make("CartPole-v0")
# video_recorder = None
# video_recorder = VideoRecorder(env=env, base_path="env_temp_video", enabled=True)

# env.reset()
# done = False
# while not done:
#     env.render()
#     video_recorder.capture_frame()
#     _, _, done, _ = env.step(env.action_space.sample())

# video_recorder.close()
# video_recorder.enabled = False
# env.close()

import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('CartPole-v0'), './video', force=True)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
env.close()