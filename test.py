import gym
from hecogrid.envs import register_marl_env, get_env_class
import cv2
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

n_agents = 30
num_goals = 100
grid_size = 30
max_steps = 200
view_size = 7
view_tile_size = 8
clutter_density = 0.1
coordination = 1
heterogeneity = 1
coordination_level = 1
heterogenetiy = 1
seed = 42

env_class_name_1 = "KeyForTreasureEnv"
env_class_name_2 = "TeamTogetherEnv"
env_class_name_3 = "TeamSupportEnv"

env_class_1 = get_env_class(env_class_name_1)
env_class_2 = get_env_class(env_class_name_2)
env_class_3 = get_env_class(env_class_name_3)

env_instance_name_1 = f"{env_class_name_1}_{n_agents}Agents_{num_goals}Goals-v0"
env_instance_name_2 = f"{env_class_name_2}_{n_agents}Agents_{num_goals}Goals-v0"
env_instance_name_3 = f"{env_class_name_3}_{n_agents}Agents_{num_goals}Goals-v0"

# Registering KeyForTreasure task
register_marl_env(
    env_instance_name_1,
    env_class_1,
    n_agents=n_agents,
    grid_size=grid_size,
    max_steps=max_steps,
    view_size=view_size,
    view_tile_size=view_tile_size,
    view_offset=1,
    seed=seed,
    env_kwargs={
        'clutter_density': clutter_density,
        'n_bonus_tiles': num_goals,
        'coordination_level': coordination_level,
        'heterogeneity': heterogeneity,
        'n_keys': 30,
    }
)

# Registering TeamTogether task
register_marl_env(
    env_instance_name_2,
    env_class_2,
    n_agents=n_agents,
    grid_size=grid_size,
    max_steps=max_steps,
    view_size=view_size,
    view_tile_size=view_tile_size,
    view_offset=1,
    seed=seed,
    env_kwargs={
        'clutter_density': clutter_density,
        'n_bonus_tiles': num_goals,
        'coordination_level': coordination_level,
        'heterogeneity': heterogeneity,
    }
)

# Registering TeamSupport task
register_marl_env(
    env_instance_name_3,
    env_class_3,
    n_agents=n_agents,
    grid_size=grid_size,
    max_steps=max_steps,
    view_size=view_size,
    view_tile_size=view_tile_size,
    view_offset=1,
    seed=seed,
    env_kwargs={
        'clutter_density': clutter_density,
        'n_bonus_tiles': num_goals,
        'coordination_level': coordination_level,
        'heterogeneity': heterogeneity,
    }
)

env = gym.make(env_instance_name_1)
img = env.grid.render(tile_size=100)
cv2.imwrite(f'{env_class_name_1}.png', img[:,:,[2,1,0]])

env = gym.make(env_instance_name_2)
img = env.grid.render(tile_size=100)
cv2.imwrite(f'{env_class_name_2}.png', img[:,:,[2,1,0]])

env = gym.make(env_instance_name_3)
img = env.grid.render(tile_size=100)
cv2.imwrite(f'{env_class_name_3}.png', img[:,:,[2,1,0]])