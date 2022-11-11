from ..base import MultiGridEnv, MultiGrid
from ..objects import *
import itertools
from random import shuffle

class TeamTogetherEnv(MultiGridEnv):
    mission = "collect as many treasures as possible"
    metadata = {}

    def __init__(self, *args, reward=1, penalty=0.0, n_clutter=None, clutter_density=None, n_bonus_tiles=3, initial_reward=True, reward_decay=False, coordination_level=1, heterogeneity=1, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        self.heterogeneity = heterogeneity
        self.coordination_level = coordination_level

        # List of colors

        self.COLORS = {
            "red": np.array([255, 0, 0]),
            "orange": np.array([255, 165, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
            "cyan": np.array([0, 139, 139]),
            "purple": np.array([112, 39, 195]),
            "yellow": np.array([255, 255, 0]),
            "custom1": np.array([0, 255, 255]),
            "custom2": np.array([0, 128, 128]),
            "custom3": np.array([39, 112, 195]),
            "custom4": np.array([192, 195, 112]),
            "custom5": np.array([95, 255, 128]),
            "custom6": np.array([139, 128, 255]),
            "olive": np.array([128, 128, 0]),
            "grey": np.array([100, 100, 100]),
            "worst": np.array([74, 65, 42]),  # https://en.wikipedia.org/wiki/Pantone_448_C
            "pink": np.array([255, 0, 189]),
            "white": np.array([255,255,255]),
            "prestige": np.array([255,255,255]),
            "shadow": np.array([35,25,30]), # nice dark purpley color for cells agents can't see.
            "black": np.array([0, 0, 0])
        }

        # Different permutation of actions , for different zone 
        self.AllActions = []
        AllActions = list(itertools.permutations(list(range(7)))) # it is import that the SAME order is used across episodes

        indices = list(range(0, len(AllActions), 500))
        self.AllActions = [AllActions[i] for i in indices]

        # Overwrite the default reward_decay for goal cycle environments.
        super().__init__(*args, **{**kwargs, 'reward_decay': reward_decay})

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter
        
        self.reward = reward
        self.penalty = penalty

        self.initial_reward = initial_reward
        self.n_bonus_tiles = n_bonus_tiles
        self.bonus_tiles = []

        # Overwrite the default reward_decay for goal cycle environments.
        super().__init__(*args, **{**kwargs, 'reward_decay': reward_decay})

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        for bonus_id in range(getattr(self, 'n_bonus_tiles', 0)):
            self.place_obj(
                GoalTeam(
                    color="yellow",
                    reward=1,
                    coordination=self.coordination_level,
                ),
                max_tries=100
            )
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)

    def step(self, actions):
        # Spawn agents if it's time.
        for agent in self.agents:
            if not agent.active and not agent.done and self.step_count >= agent.spawn_delay:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()
              
        for agent in self.agents:
            #which zone the agent is currently in , zone determined by horizonal position
            ZoneIndex=int((agent.pos[0]//(self.width//self.heterogeneity)))
            agent.ZoneIndex=ZoneIndex
            #change the color of the agent depending on which zone it is in, so that it knows what policy to use
            agent.color=list(self.COLORS.keys())[ZoneIndex]
            
            ###change the action scheme of tha agent depending to which zone the agent is in
            ActionSet=self.AllActions[ZoneIndex]

            class actions_(IntEnum):
                left = ActionSet[0]  # Rotate left
                right = ActionSet[1]  # Rotate right
                forward = ActionSet[2]  # Move forward
                pickup = ActionSet[3]  # Pick up an object
                drop = ActionSet[4]  # Drop an object
                toggle = ActionSet[5]  # Toggle/activate an object
                done = ActionSet[6]  # Done completing task

            agent.actions=actions_

        assert len(actions) == len(self.agents)

        step_rewards = np.zeros((len(self.agents,)), dtype=np.float)

        self.step_count += 1

        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        self.np_random.shuffle(iter_order)
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                # Rotate left
                if action == agent.actions.left:
                    agent.dir = (agent.dir - 1) % 4

                # Rotate right
                elif action == agent.actions.right:
                    agent.dir = (agent.dir + 1) % 4

                # Move forward
                elif action == agent.actions.forward:
                    # Under the follow conditions, the agent can move forward.
                    can_move = fwd_cell is None or fwd_cell.can_overlap()
                    if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                        can_move = False

                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        else:
                            assert cur_cell.can_overlap()
                            cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else: # How was "agent" there in teh first place?
                                raise ValueError("?!?!?!")

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = [] 
                        # test_integrity(f"After moving {agent.color} fellow")

                        # Rewards can be got iff. fwd_cell has a "get_reward" method


                        if hasattr(fwd_cell, 'get_reward'):
                            rwd = fwd_cell.get_reward(agent)
                            if bool(self.reward_decay):
                                rwd *= (1.0-0.9*(self.step_count/self.max_steps))
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)
                            

                        if isinstance(fwd_cell, (Lava, Goal)):
                            agent.done = True

                # TODO: verify pickup/drop/toggle logic in an environment that 
                #  supports the relevant interactions.
                # Pick up an object
                elif action == agent.actions.pickup:
                    if hasattr(cur_cell, 'get_reward'):
                        rwd = cur_cell.get_reward(agent)
                        if bool(self.reward_decay):
                            rwd *= (1.0-0.9*(self.step_count/self.max_steps))
                        step_rewards[agent_no] += rwd
                        agent.reward(rwd)
                    
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                    else:
                        pass

                # Done action (not used by default)
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(f"Environment can't handle action {action}.")

                agent.on_step(fwd_cell if agent_moved else None)

        
        # If any of the agents individually are "done" (hit lava or in some cases a goal) 
        #   but the env requires respawning, then respawn those agents.
        for agent in self.agents:
            if agent.done:
                if self.respawn:
                    resting_place_obj = self.grid.get(*agent.pos)
                    if resting_place_obj == agent:
                        if agent.agents:
                            self.grid.set(*agent.pos, agent.agents[0])
                            agent.agents[0].agents += agent.agents[1:]
                        else:
                            self.grid.set(*agent.pos, None)
                    else:
                        resting_place_obj.agents.remove(agent)
                        resting_place_obj.agents += agent.agents[:]
                        agent.agents = []
                        
                    agent.reset(new_episode=False)
                    self.place_obj(agent, **self.agent_spawn_kwargs)
                    agent.activate()
                else: # if the agent shouldn't be respawned, then deactivate it.
                    agent.deactivate()

        # The episode overall is done if all the agents are done, or if it exceeds the step limit.
        done = (self.step_count >= self.max_steps) or all([agent.done for agent in self.agents])
        info = {f'{i}': agent.done for i, agent in enumerate(self.agents)}

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        # To make the env fully cooperative, all agent share identical reward 
        step_rewards[step_rewards >= step_rewards.min()] = step_rewards.mean()
        return obs, step_rewards, done, info