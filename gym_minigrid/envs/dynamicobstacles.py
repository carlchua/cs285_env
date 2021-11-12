from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add

class DynamicObstaclesEnv(MiniGridEnv):
    """
    Single-room square grid environment with moving obstacles
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            n_obstacles=4
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.move = False

        # Reduce obstacles if there are too many
        self.n_obstacles = int(n_obstacles)
        # if n_obstacles <= size/2 + 1:
        #     self.n_obstacles = int(n_obstacles)
        # else:
        #     self.n_obstacles = int(size/2)
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], top=(2,2), size=(self.grid.width-4,self.grid.height-4), max_tries=100)

        self.mission = "View Every Tile"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'goal'

        # Update obstacle positions
        if self.move:
            for i_obst in range(len(self.obstacles)):
                old_pos = self.obstacles[i_obst].cur_pos
                top = tuple(map(add, old_pos, (1, 0)))

                try:
                    new_pos = self.place_obj(self.obstacles[i_obst], top=top, size=(1,1), max_tries=100)
                    self.grid.set(*old_pos, None)

                    # If obstacle is at edge, remove and respawn at the side
                    if new_pos[1] >= self.grid.width-2 or new_pos[0] >= self.grid.height-2:
                        # print("hello")
                        self.place_obj(self.obstacles[i_obst], top=(1,1), size=(1,self.grid.height-3), max_tries=100)
                        self.grid.set(*new_pos, None)
                except:
                    pass
            self.move = False
        elif not self.move:
            self.move = True


        # Update the agent's position/direction
        obs, reward, done, info = MiniGridEnv.step(self, action)

        # If the agent walked over an obstacle or wall
        if action == self.actions.forward and not_clear:
            # PENALTY FOR COLLISION
            reward = -20
            return obs, reward, done, info

        return obs, reward, done, info

class DynamicObstaclesEnv5x5(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=5, n_obstacles=2)

class DynamicObstaclesRandomEnv5x5(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None, n_obstacles=2)

class DynamicObstaclesEnv6x6(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=3)

class DynamicObstaclesRandomEnv6x6(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_obstacles=3)

class DynamicObstaclesEnv16x16(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=8)

class DynamicObstaclesEnv24x24(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=24, n_obstacles=25)

register(
    id='MiniGrid-Dynamic-Obstacles-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-8x8-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv'
)

register(
    id='MiniGrid-Dynamic-Obstacles-16x16-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv16x16'
)

register(
    id='MiniGrid-Dynamic-Obstacles-24x24-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv24x24'
)
