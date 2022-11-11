class MultiGridEnv(gym.Env):
    def __init__(
        self,
        agents = [],
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        reward_decay=True,
        seed=1337,
        respawn=False,
        ghost_mode=True,

        agent_spawn_kwargs = {}
    ):

        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        self.respawn = respawn

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reward_decay = reward_decay
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode

        self.agents = []
        for agent in agents:
            self.add_agent(agent)

        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        return gym.spaces.Tuple(
            [agent.action_space for agent in self.agents]
        )

    @property
    def observation_space(self):
        return gym.spaces.Tuple(
            [agent.observation_space for agent in self.agents]
        )

    @property
    def num_agents(self):
        return len(self.agents)
    
    def add_agent(self, agent_interface):
        if isinstance(agent_interface, dict):
            self.agents.append(GridAgentInterface(**agent_interface))
        elif isinstance(agent_interface, GridAgentInterface):
            self.agents.append(agent_interface)
        else:
            raise ValueError(
                "To add an agent to a marlgrid environment, call add_agent with either a GridAgentInterface object "
                " or a dictionary that can be used to initialize one.")

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            if agent.spawn_delay == 0:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        self.step_count = 0
        obs = self.gen_obs()
        return obs

    def gen_obs_grid(self, agent):
        # If the agent is inactive, return an empty grid and a visibility mask that hides everything.
        if not agent.active:
            # below, not sure orientation is correct but as of 6/27/2020 that doesn't matter because
            # agent views are usually square and this grid won't be used for anything.
            grid = MultiGrid((agent.view_size, agent.view_size), orientation=agent.dir+1)
            vis_mask = np.zeros((agent.view_size, agent.view_size), dtype=np.bool)
            return grid, vis_mask

        topX, topY, botX, botY = agent.get_view_exts()

        grid = self.grid.slice(
            topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
        )

        # Process occluders and visibility
        # Note that this incurs some slight performance cost
        vis_mask = agent.process_vis(grid.opacity)

        # Warning about the rest of the function:
        #  Allows masking away objects that the agent isn't supposed to see.
        #  But breaks consistency between the states of the grid objects in the parial views
        #   and the grid objects overall.
        if len(getattr(agent, 'hide_item_types', []))>0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i,j)
                    if (item is not None) and (item is not agent) and (item.type in agent.hide_item_types):
                        if len(item.agents) > 0:
                            grid.set(i,j,item.agents[0])
                        else:
                            grid.set(i,j,None)

        return grid, vis_mask

    def gen_agent_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid(agent)
        grid_image = grid.render(tile_size=agent.view_tile_size, visible_mask=vis_mask, top_agent=agent)
        if agent.observation_style=='image':
            return grid_image
        else:
            ret = {'pov': grid_image}
            if agent.observe_rewards:
                ret['reward'] = getattr(agent, 'step_reward', 0)
            if agent.observe_position:
                agent_pos = agent.pos if agent.pos is not None else (0,0)
                ret['position'] = np.array(agent_pos)/np.array([self.width, self.height], dtype=np.float)
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret['orientation'] = agent_dir
            return ret

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in self.agents]

    def __str__(self):
        return self.grid.__str__()

    def check_agent_position_integrity(self, title=''):
        '''
        This function checks whether each agent is present in the grid in exactly one place.
        This is particularly helpful for validating the world state when ghost_mode=False and
        agents can stack, since the logic for moving them around gets a bit messy.
        Prints a message and drops into pdb if there's an inconsistency.
        '''
        agent_locs = [[] for _ in range(len(self.agents))]
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                x = self.grid.get(i,j)
                for k,agent in enumerate(self.agents):
                    if x==agent:
                        agent_locs[k].append(('top', (i,j)))
                    if hasattr(x, 'agents') and agent in x.agents:
                        agent_locs[k].append(('stacked', (i,j)))
        if not all([len(x)==1 for x in agent_locs]):
            print(f"{title} > Failed integrity test!")
            for a, al in zip(self.agents, agent_locs):
                print(" > ", a.color,'-', al)
            import pdb; pdb.set_trace()

    def step(self, actions):
        # Spawn agents if it's time.
        for agent in self.agents:
            if not agent.active and not agent.done and self.step_count >= agent.spawn_delay:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()



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

        return obs, step_rewards, done, info

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid. Replace anything that is already there.
        """
        self.grid.set(i, j, obj)
        if obj is not None:
            obj.set_position((i,j))
        return True

    def try_place_obj(self,obj, pos):
        ''' Try to place an object at a certain position in the grid.
        If it is possible, then do so and return True.
        Otherwise do nothing and return False. '''
        # grid_obj: whatever object is already at pos.
        grid_obj = self.grid.get(*pos)

        # If the target position is empty, then the object can always be placed.
        if grid_obj is None:
            self.grid.set(*pos, obj)
            obj.set_position(pos)
            return True

        # Otherwise only agents can be placed, and only if the target position can_overlap.
        if not (grid_obj.can_overlap() and obj.is_agent):
            return False

        # If ghost mode is off and there's already an agent at the target cell, the agent can't
        #   be placed there.
        if (not self.ghost_mode) and (grid_obj.is_agent or (len(grid_obj.agents)>0)):
            return False

        grid_obj.agents.append(obj)
        obj.set_position(pos)
        return True

    def place_obj(self, obj, top=(0,0), size=None, reject_fn=None, max_tries=1e5):
        max_tries = int(max(1, min(max_tries, 1e5)))
        top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        bottom = (min(top[0] + size[0], self.grid.width), min(top[1] + size[1], self.grid.height))

        # agent_positions = [tuple(agent.pos) if agent.pos is not None else None for agent in self.agents]
        for try_no in range(max_tries):
            pos = self.np_random.randint(top, bottom)
            if (reject_fn is not None) and reject_fn(pos):
                continue
            else:
                if self.try_place_obj(obj, pos):
                    break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        return pos

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=1000):
        # warnings.warn("Placing agents with the function place_agents is deprecated.")
        pass

    def render(
        self,
        mode="human",
        close=False,
        highlight=True,
        tile_size=TILE_PIXELS,
        show_agent_views=True,
        max_agents_per_col=3,
        agent_col_width_frac = 0.3,
        agent_col_padding_px = 2,
        pad_grey = 100
    ):
        """
        Render the whole-grid human view
        """
        pass

        # if close:
        #     if self.window:
        #         self.window.close()
        #     return

        # if mode == "human" and not self.window:
        #     # from gym.envs.classic_control.rendering import SimpleImageViewer

        #     self.window = SimpleImageViewer(caption="Marlgrid")

        # # Compute which cells are visible to the agent
        # highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        # for agent in self.agents:
        #     if agent.active:
        #         xlow, ylow, xhigh, yhigh = agent.get_view_exts()
        #         dxlow, dylow = max(0, 0-xlow), max(0, 0-ylow)
        #         dxhigh, dyhigh = max(0, xhigh-self.grid.width), max(0, yhigh-self.grid.height)
        #         if agent.see_through_walls:
        #             highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] = True
        #         else:
        #             a,b = self.gen_obs_grid(agent)
        #             highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] |= (
        #                 rotate_grid(b, a.orientation)[dxlow:(xhigh-xlow)-dxhigh, dylow:(yhigh-ylow)-dyhigh]
        #             )


        # # Render the whole grid
        # img = self.grid.render(
        #     tile_size, highlight_mask=highlight_mask if highlight else None
        # )
        # rescale = lambda X, rescale_factor=2: np.kron(
        #     X, np.ones((int(rescale_factor), int(rescale_factor), 1))
        # )

        # if show_agent_views:

        #     target_partial_width = int(img.shape[0]*agent_col_width_frac-2*agent_col_padding_px)
        #     target_partial_height = (img.shape[1]-2*agent_col_padding_px)//max_agents_per_col

        #     agent_views = [self.gen_agent_obs(agent) for agent in self.agents]
        #     agent_views = [view['pov'] if isinstance(view, dict) else view for view in agent_views]
        #     agent_views = [rescale(view, min(target_partial_width/view.shape[0], target_partial_height/view.shape[1])) for view in agent_views]
        #     # import pdb; pdb.set_trace()
        #     agent_views = [agent_views[pos:pos+max_agents_per_col] for pos in range(0, len(agent_views), max_agents_per_col)]

        #     f_offset = lambda view: np.array([target_partial_height - view.shape[1], target_partial_width - view.shape[0]])//2
            
        #     cols = []
        #     for col_views in agent_views:
        #         col = np.full(( img.shape[0],target_partial_width+2*agent_col_padding_px,3), pad_grey, dtype=np.uint8)
        #         for k, view in enumerate(col_views):
        #             offset = f_offset(view) + agent_col_padding_px
        #             offset[0] += k*target_partial_height
        #             col[offset[0]:offset[0]+view.shape[0], offset[1]:offset[1]+view.shape[1],:] = view
        #         cols.append(col)

        #     img = np.concatenate((img, *cols), axis=1)

        # if mode == "human":
        #     if not self.window.isopen:
        #         self.window.imshow(img)
        #         self.window.window.set_caption("Marlgrid")
        #     else:
        #         self.window.imshow(img)

        # return img