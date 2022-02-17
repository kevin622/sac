import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def video_record(ith_episode, env, env_name, agent):
    if not os.path.exists("video/"):
        os.makedirs('video/')
    video_recorder = None
    video_recorder = VideoRecorder(env=env,
                                   base_path=f'video/{env_name}_{ith_episode}',
                                   enabled=True)
    state = env.reset()
    done = False
    while not done:
        env.render(mode='rgb_array')
        video_recorder.capture_frame()
        action = agent.get_mean_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
    video_recorder.close()
    video_recorder.enabled = False