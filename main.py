import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from copy import deepcopy
from cbs_solver import ConflictBasedSearch, Agent

######### HELPER FUNCTIONS TO SET UP ENVIRONMENT WITH OBSTACLES #########

@dataclass
class Obstacle: # only convex allowed here
    vertices: List[Tuple[float, float]]

# check if point is inside a convex obstacle
# uses cross product method
def point_in_convex_polygon(point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> bool:
    x, y = point
    n = len(vertices)
    
    # to be outside the obstacle, the point has to be on the outside of an edge
    sign = None
    for i in range(n):
        # grabbing a single edge with 2 vertices
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        
        # Cross product of edge and point, the sign is what side the point is on
        # kind like having primatives
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        
        if abs(cross_product) < 1e-10:  # it's on the edge, just ignore
            continue
            
        if sign is None:
            sign = cross_product > 0
        elif (cross_product > 0) != sign:
            return False # point is on a different side of this edge, so outside
    
    return True # point is inside all edges

def create_grid_from_obstacles(workspace_width: float, workspace_height: float, 
                              resolution: float, obstacles: List[Obstacle]) -> Tuple[List[List[int]], float]:
    # makes grid of 1s and 0s representing obstacles from convex polygons
 
    grid_width = int(np.ceil(workspace_width / resolution))
    grid_height = int(np.ceil(workspace_height / resolution))
    
    # actual cell size
    actual_cell_size_x = workspace_width / grid_width
    actual_cell_size_y = workspace_height / grid_height
    
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    
    for row in range(grid_height):
        for col in range(grid_width):
            # check for obstacle with cell center in world coordinates
            cell_center_x = (col + 0.5) * actual_cell_size_x
            cell_center_y = (row + 0.5) * actual_cell_size_y
            
            for obstacle in obstacles:
                if point_in_convex_polygon((cell_center_x, cell_center_y), obstacle.vertices):
                    grid[row][col] = 1
                    break

    return grid, actual_cell_size_x

# this is the "prison" layout. Doesn't really look like a prison at all haha
def create_sample_problem_with_obstacles():

    workspace_width = 20.0
    workspace_height = 15.0
    
    resolution = 0.25  # each grid cell is this tall and wide
    
    obstacles = [ # in world coordinates
        Obstacle(vertices=[(2.0, 0), (6.0, 0), (6.0, 4.0), (2.0, 4.0)]),
        Obstacle(vertices=[(8.0, 6.0), (12.0, 0.5), (10.0, 12.0)]),
        Obstacle(vertices=[(15.0, 2.0), (18.0, 2.0), (18.0, 5.0), (15.0, 5.0)]),
        Obstacle(vertices=[(4.0, 9.0), (7.0, 8.0), (5.0, 11.5)])
    ]
    
    # discritize workspace into grid
    grid, cell_size = create_grid_from_obstacles(workspace_width, workspace_height, 
                                                resolution, obstacles)
    
    # just helper to convert goal and start positions from world coords
    def world_to_grid(world_pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = world_pos
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        return (grid_y, grid_x)  # row, col
    
    agents = [
        Agent(id=0, start=world_to_grid((1.0, 1.0)), goal=world_to_grid((18.0, 13.0))),
        Agent(id=1, start=world_to_grid((18.0, 1.0)), goal=world_to_grid((1.0, 13.0))),
        Agent(id=2, start=world_to_grid((15.0, 1.0)), goal=world_to_grid((1.0, 1.0))),
        Agent(id=3, start=world_to_grid((18.0, 10.0)), goal=world_to_grid((15.0, 1.0))),
    ]

    ## added these checks for debugging, when I changed start/goal positions of agents to impossible positions

    # check conflicts
    for a in agents:
        for b in agents:
            if a.id >= b.id:
                continue
            if a.start == b.start or a.goal == b.goal:
                print(f"Conflict in init positions or goals: Agent {a.id}, Agent {b.id}")

    # check grid validity
    for agent in agents:
        sy, sx = agent.start
        gy, gx = agent.goal
        if grid[sy][sx] == 1:
            print(f"Agent {agent.id} start pos inside obstacle")
        if grid[gy][gx] == 1:
            print(f"Agent {agent.id} goal pos inside obstacle")
    
    return grid, agents, obstacles, workspace_width, workspace_height, cell_size

class InteractivePlanner:
    """Interactive planner that allows changing agent 1's goal with arrow keys."""
    
    def __init__(self, grid, agents, obstacles, workspace_width, workspace_height, cell_size):
        self.grid = grid
        self.agents = agents.copy()  # Make a copy so we can modify it
        self.obstacles = obstacles
        self.workspace_width = workspace_width
        self.workspace_height = workspace_height
        self.cell_size = cell_size
        self.cbs = ConflictBasedSearch(grid, self.agents)

        # Game state variables
        self.current_solution = None
        self.current_timestep = 0
        self.game_over = False
        self.animation_timer = None
        
        # Set up the figure (smaller size for faster rendering)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initialize containers for dynamic elements
        self.agent_current_positions = {}  # Current positions of agents
        self.agent_position_objects = {}   # Visual objects for current positions
        self.detection_circles = []  # Visual objects for detection radius circles
        self.no_solution_text = None

        # toggle radius where guards can 'see' the prisoner
        self.detection_radius = 25
        self.prisoner_found = False
        self.last_prisoner_found = False
        self.guard_quadrants = self._assign_quadrants()
        # Assign coverage goals to guards
        self.coverage_goals = {}
        for agent in self.agents:
            self.coverage_goals[agent.id] = self.generate_coverage_goals(self.guard_quadrants[agent.id])
        self.coverage_goal_indeces = {agent.id: 0 for agent in self.agents}
        self.door_loc = self.agents[2].goal  # prisoner exit door location
        self.prisoner_loc = self.agents[1].goal  # prisoner start location
        self.guard_two_to_door = False  # flag to indicate if guard 2 is heading to door

        # Draw static background once
        self.setup_static_background()

        # self.agents[0].goal = (self.agents[1].goal[0] + 1, self.agents[1].goal[1] + 1)  # offset to avoid overlap
        # self.agents[3].goal = (self.agents[1].goal[0] - 1, self.agents[1].goal[1] - 1)  # offset to avoid overlap

        # start in patrol mode, not chasing prisoner
        self.update_patrol_goals()
        
        # Calculate and plot initial solution
        self.calculate_new_solution()

    def _assign_quadrants(self):
        """Assign each guard to a quadrant of the workspace."""
        rows = len(self.grid)
        cols = len(self.grid[0])
        
        # Define quadrants: (row_start, row_end, col_start, col_end)
        quadrants = {
            0: (0, rows//2, 0, cols//2),           # Top-left
            2: (rows//2, rows, 0, cols//2),        # Bottom-left  
            3: (0, rows//2, cols//2, cols),        # Top-right
            1: (rows//2, rows, cols//2, cols)      # Bottom-right
        }
        return quadrants
    
    def generate_coverage_goals(self, quadrant: Tuple[int, int, int, int], padding: int = 5) -> List[Tuple[int, int]]:
        """Generate a simple coverage path for a guard in its quadrant."""
        row_start, row_end, col_start, col_end = quadrant
        goals = []
        
        last_point = None
        for r in range(row_start, row_end, padding):
            keep_next_point = True
            if (r - row_start) % (2 * padding) == 0:
                # Left to right
                for c in range(col_start, col_end, padding):
                    if self.grid[r][c] == 0:
                        last_point = (r, c)
                        if keep_next_point:
                            goals.append((r, c))
                            keep_next_point = False
                    else:
                        if last_point is not None:
                            goals.append(last_point)
                        keep_next_point = True
            else:
                # Right to left
                for c in range(col_end - 1, col_start - 1, -padding):
                    if self.grid[r][c] == 0:
                        last_point = (r, c)
                        if keep_next_point:
                            goals.append((r, c))
                            keep_next_point = False
                    else:
                        if last_point is not None:
                            goals.append(last_point)
                        keep_next_point = True
            goals.append(last_point)  # Ensure we end the row at last valid point
        return goals
    
    def plot_coverage_goals(self):
        """Plot coverage goals for each guard."""
        # Define colors for each agent's patrol path
        patrol_colors = {
            0: 'cyan',
            1: 'orange', 
            2: 'lime',
            3: 'magenta'
        }
        
        for agent in self.agents:
            goals = self.coverage_goals[agent.id]
            if not goals:
                continue
                
            goal_world_coords = [self.grid_to_world(g) for g in goals]
            xs, ys = zip(*goal_world_coords)
            
            color = patrol_colors.get(agent.id, 'gray')
            self.ax.plot(xs, ys, linestyle='--', marker='o', markersize=3,
                        color=color, alpha=0.5, linewidth=1.5,
                        label=f'Agent {agent.id} Patrol Path')
        
        self.ax.legend(loc='upper right', fontsize=8)
    
    def _check_prisoner_detection(self):
        """Check if any guard can detect the prisoner."""
        
        agents_seeing_prisoner = []
        for agent in self.agents:
            dx = self.prisoner_loc[0] - self.agent_current_positions[agent.id][0]
            dy = self.prisoner_loc[1] - self.agent_current_positions[agent.id][1]
            distance = (dx**2 + dy**2)**0.5  # sqrt(dx^2 + dy^2)
            if distance <= self.detection_radius:
                agents_seeing_prisoner.append(agent.id)

        if agents_seeing_prisoner == [2]:
            self.guard_two_to_door = False # only guard 2 sees prisoner, so have them follow
        elif len(agents_seeing_prisoner) > 0:
            self.guard_two_to_door = True  # other guards see prisoner, have guard 2 go to door 
            if self.agents[2].goal != self.door_loc:
                self.agents[2].goal = self.door_loc  # set guard 2's goal to door location  
                self.calculate_new_solution()
        return agents_seeing_prisoner
    
    def update_patrol_goals(self, agent_id: Optional[int] = None):
        if agent_id is not None:
            agent = next(a for a in self.agents if a.id == agent_id)
            coverage_goals = self.coverage_goals[agent.id]
            agent.goal = coverage_goals[self.coverage_goal_indeces[agent.id] % len(coverage_goals)]
            self.coverage_goal_indeces[agent.id] += 1
            return

        for agent in self.agents:
            coverage_goals = self.coverage_goals[agent.id]
            agent.goal = coverage_goals[self.coverage_goal_indeces[agent.id] % len(coverage_goals)]
            self.coverage_goal_indeces[agent.id] += 1

    def world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x, y = world_pos
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        # Clamp to grid bounds
        grid_x = max(0, min(len(self.grid[0]) - 1, grid_x))
        grid_y = max(0, min(len(self.grid) - 1, grid_y))
        return (grid_y, grid_x)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (center of cell)."""
        row, col = grid_pos
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return (x, y)
    
    def is_valid_goal(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if a grid position is valid for placing a goal."""
        row, col = grid_pos
    
        # Check center cell and all 8 surrounding cells (3x3 footprint)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                check_row, check_col = row + dr, col + dc
                
                # Check bounds
                if not (0 <= check_row < len(self.grid) and 
                        0 <= check_col < len(self.grid[0])):
                    return False
                
                # Check obstacle
                if self.grid[check_row][check_col] == 1:
                    return False
        
        return True
    
    def is_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if the straight-line path between start and end is obstacle-free using interpolation."""
        row1, col1 = start
        row2, col2 = end
        
        # Calculate steps needed (use max distance to ensure we check all cells)
        dr = abs(row2 - row1)
        dc = abs(col2 - col1)
        steps = max(dr, dc)
        
        if steps == 0:
            return self.is_valid_goal(end)
        
        # Interpolate along the line
        for i in range(steps + 1):
            t = i / steps
            check_row = int(row1 + t * (row2 - row1))
            check_col = int(col1 + t * (col2 - col1))
            
            # Check if this interpolated position is valid (3x3 footprint)
            if not self.is_valid_goal((check_row, check_col)):
                return False
        
        return True
    
    def on_key_press(self, event):
        """Handle keyboard input to move agent 1's goal."""
        if self.game_over:
            return
        
        if event.key not in ['up', 'down', 'left', 'right']:
            return
            
        # Get current goal of agent 1
        agent1 = next(a for a in self.agents if a.id == 1)
        current_goal = self.prisoner_loc
        
        # Calculate new goal position
        row, col = current_goal
        speed = 4
        if event.key == 'up':
            new_goal = (row - speed, col)
        elif event.key == 'down':
            new_goal = (row + speed, col)
        elif event.key == 'left':
            new_goal = (row, col - speed)
        elif event.key == 'right':
            new_goal = (row, col + speed)

        if not self.is_valid_goal(new_goal):
            print(f"Trying to move to valid goal position: {new_goal}")
            for step in range(speed-1, 0, -1):
                if event.key == 'up':
                    new_goal = (row - step, col)
                elif event.key == 'down':
                    new_goal = (row + step, col)
                elif event.key == 'left':
                    new_goal = (row, col - step)
                elif event.key == 'right':
                    new_goal = (row, col + step)
                if self.is_valid_goal(new_goal):
                    break
        
        # Check if new position is valid
        if self.is_valid_goal(new_goal) and self.is_path_clear(current_goal, new_goal):
            self.prisoner_loc = new_goal  # update prisoner location
            if self.prisoner_found:
                agent1.goal = new_goal
                agent0 = next(a for a in self.agents if a.id == 0)
                agent2 = next(a for a in self.agents if a.id == 2)
                agent3 = next(a for a in self.agents if a.id == 3)
                agent0.goal = (new_goal[0] + 1, new_goal[1] + 1)  # sync agent 0's goal if toggled, but offset to avoid overlap
                if self.guard_two_to_door:
                    agent2.goal = self.door_loc  # prisoner exit door location
                else:
                    agent2.goal = (new_goal[0] + 1, new_goal[1] - 1)  # get to follow prisoner until another guard reaches them
                agent3.goal = (new_goal[0] - 1, new_goal[1] - 1)  # sync agent 3's goal if toggled, but offset to avoid overlap

                self.draw_goal_marker()
                self.calculate_new_solution()
        else:
            print(f"Invalid goal position: {new_goal}")
    
    def calculate_new_solution(self, inc_step = True):
        """Calculate a new solution when goal changes."""
        import time
        start_time = time.time()
        
        # Update agent start positions to their current positions
        if hasattr(self, 'agent_current_positions') and self.agent_current_positions:
            # Update each agent's start position to their current position
            for agent in self.agents:
                if agent.id in self.agent_current_positions:
                    agent.start = self.agent_current_positions[agent.id]
        
        # Update the CBS with new agent configuration
        prev_solution = self.current_solution
        self.cbs = ConflictBasedSearch(self.grid, self.agents)
        solution = self.cbs.solve(previous_solution=prev_solution)
        
        calc_time = time.time() - start_time
        print(f"New path calculated in {calc_time:.20f}s")
        
        # Stop any existing animation
        if self.animation_timer:
            self.animation_timer.event_source.stop()
        
        if solution is None:
            self.show_no_solution()
            self.game_over = True
        else:
            self.current_solution = solution
            self.current_timestep = 0 if inc_step else -1
            self.game_over = False
            
            # Initialize agent positions to start positions (which are now current positions)
            for agent_id, path in solution.items():
                self.agent_current_positions[agent_id] = path[0]
            
            # Start the animation
            # Clear dynamic elements
            self.clear_dynamic_elements()
            self.start_animation()
        
        # Update display
        self.fig.canvas.draw_idle()

    def start_animation(self):
        """Start the timer for agent movement."""
        from matplotlib.animation import FuncAnimation
        
        if self.game_over or not self.current_solution:
            return
        
        # Create timer that calls update_agent_positions every X ms
        self.animation_timer = FuncAnimation(self.fig, self.update_agent_positions, 
                                        interval=200, repeat=True, blit=False)

    def update_agent_positions(self, frame):
        """Move agents one step forward in their paths."""
        if self.game_over or not self.current_solution:
            return
        
        self.current_timestep += 1
        
        # Update each agent's position
        all_at_goal = True
        agents_at_goal = []
        for agent_id, path in self.current_solution.items():
            if self.current_timestep < len(path):
                self.agent_current_positions[agent_id] = path[self.current_timestep]
                all_at_goal = False
            else:
                # Agent has reached the end of path, stay at goal
                agents_at_goal.append(agent_id)
                self.agent_current_positions[agent_id] = path[-1]
        
        # Update the visual representation
        self.update_agent_visuals()
        self.draw_goal_marker()
        
        # Check win/lose conditions
        if all_at_goal and self.prisoner_found:
            if self.agent_current_positions[2] == self.door_loc and self.agent_current_positions[1] == self.prisoner_loc and self.agent_current_positions[0] == (self.prisoner_loc[0] + 1, self.prisoner_loc[1] + 1) and self.agent_current_positions[3] == (self.prisoner_loc[0] -1, self.prisoner_loc[1] -1):
                self.end_game("GAME OVER")
                print("All agents reached their goals.")

        if agents_at_goal and not self.prisoner_found:
            # If any guard reached their patrol goal, update patrol goals and recalculate paths
            for agent_id in agents_at_goal:
                print(f"Guard {agent_id} reached patrol goal, updating patrol paths.")
                self.update_patrol_goals(agent_id)  # continue moving guards in patrol
            self.calculate_new_solution(inc_step=False)  # recalculate paths for patrol

        self.prisoner_found = self._check_prisoner_detection()
        if not self.last_prisoner_found == self.prisoner_found:
            # initialize chase mode
            if self.prisoner_found and not self.last_prisoner_found:
                print("----------------Prisoner detected! Guards are chasing!")
                agent1 = next(a for a in self.agents if a.id == 1)
                agent0 = next(a for a in self.agents if a.id == 0)
                agent2 = next(a for a in self.agents if a.id == 2)
                agent3 = next(a for a in self.agents if a.id == 3)
                agent1.goal = self.prisoner_loc
                agent0.goal = (self.prisoner_loc[0] + 1, self.prisoner_loc[1] + 1)  # offset to avoid overlap
                if self.guard_two_to_door:
                    agent2.goal = self.door_loc  # prisoner exit door location
                else:
                    agent2.goal = (self.prisoner_loc[0] + 1, self.prisoner_loc[1] - 1)  # offset to avoid overlap
                agent3.goal = (self.prisoner_loc[0] - 1, self.prisoner_loc[1] - 1)  # offset to avoid overlap
                self.calculate_new_solution(inc_step=False)
            if not self.prisoner_found and self.last_prisoner_found:
                print("----Prisoner lost! Guards resuming patrol.")
                self.update_patrol_goals()
                self.calculate_new_solution(inc_step=False)

        self.last_prisoner_found = self.prisoner_found

    def draw_goal_marker(self):
        """Draw the goal position that the user is controlling."""
        goal_pos = self.prisoner_loc
        
        # Convert to world coordinates
        world_x = (goal_pos[1] + 0.5) * self.cell_size
        world_y = (goal_pos[0] + 0.5) * self.cell_size
        
        # Remove existing goal marker if it exists
        if hasattr(self, 'goal_marker') and self.goal_marker:
            self.goal_marker.remove()
        
        # Draw goal as a large square with distinctive appearance
        self.goal_marker = self.ax.scatter(world_x, world_y, c='gold', s=400, 
                                        marker='s', edgecolors='black', linewidth=3,
                                        zorder=15, alpha=0.9, label='Goal (Move with arrows)')

    def end_game(self, message):
        """End the game with a message."""
        self.game_over = True
        
        if self.animation_timer:
            self.animation_timer.event_source.stop()
        
        # Display game over message
        self.ax.text(0.5, 0.5, message, transform=self.ax.transAxes, 
                    ha='center', va='center', fontsize=20, color='red', 
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"\n{message}")
        self.fig.canvas.draw_idle()

    def setup_static_background(self):
        """Set up static elements that don't change (obstacles, grid, labels)."""
        # Plot original obstacle polygons
        if self.obstacles:
            for obstacle in self.obstacles:
                vertices = np.array(obstacle.vertices + [obstacle.vertices[0]])
                self.ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2, alpha=0.7)
                self.ax.fill(vertices[:, 0], vertices[:, 1], color='gray', alpha=0.3)
        
        # Plot grid obstacles
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == 1:
                    world_x = j * self.cell_size
                    world_y = i * self.cell_size
                    self.ax.add_patch(plt.Rectangle((world_x, world_y), self.cell_size, self.cell_size, 
                                                facecolor='red', alpha=0.4, edgecolor='red'))
        
        # self.plot_coverage_goals()

        # Set up plot appearance (this stays constant)
        self.ax.set_xlim(0, self.workspace_width)
        self.ax.set_ylim(0, self.workspace_height)
        self.ax.set_xlabel('X (world coordinates)')
        self.ax.set_ylabel('Y (world coordinates)')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.invert_yaxis()

        # Draw the initial goal marker
        self.goal_marker = None
        self.goal_text = None
        self.draw_goal_marker()
        
    def clear_dynamic_elements(self):
        """Remove only the dynamic elements (current agent positions)."""
        # Remove agent position objects
        for agent_id, pos_obj in self.agent_position_objects.items():
            if pos_obj:
                pos_obj.remove()
        self.agent_position_objects.clear()
        
        # Remove detection circles
        for circle in self.detection_circles:
            circle.remove()
        self.detection_circles.clear()
        
        # Remove no solution text if it exists
        if self.no_solution_text:
            self.no_solution_text.remove()
            self.no_solution_text = None

    def update_agent_visuals(self):
        """Update the visual representation of current agent positions."""
        # Clear existing position objects
        for agent_id, pos_obj in self.agent_position_objects.items():
            if pos_obj:
                pos_obj.remove()
        
        # Define colors for agents
        agent_colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        # Draw current positions
        for agent_id, grid_pos in self.agent_current_positions.items():
            agent_color = agent_colors[agent_id % len(agent_colors)]
            
            # Convert grid to world coordinates
            world_x = (grid_pos[1] + 0.5) * self.cell_size
            world_y = (grid_pos[0] + 0.5) * self.cell_size
            
            # Create square marker for current position
            if agent_id == 1 or agent_id == 0 or agent_id ==3 or agent_id ==4:
                # Special large circle for agent 1 (the one we control)
                marker, size, edgecolor = 'o', 100, 'black'
            else:
                # Square for other agents
                marker, size, edgecolor = 's', 100, 'black'
            
            pos_obj = self.ax.scatter(world_x, world_y, c=agent_color, s=size, 
                                    marker=marker, edgecolors=edgecolor, linewidth=3,
                                    zorder=10, alpha=0.9)
            
            self.agent_position_objects[agent_id] = pos_obj
        
        # Draw detection circles for guards in patrol mode
        self.draw_detection_circles()
        
        # Update display
        self.fig.canvas.draw_idle()
    
    def draw_detection_circles(self):
        """Draw detection radius circles around guards in patrol mode."""
        # Remove old circles
        for circle in self.detection_circles:
            circle.remove()
        self.detection_circles.clear()
        
        # Only draw circles when NOT in chase mode (patrol mode)
        if not self.prisoner_found:
            for guard_id in [0, 1, 2, 3]:  # Guard IDs
                if guard_id in self.agent_current_positions:
                    guard_pos = self.agent_current_positions[guard_id]
                    world_x = (guard_pos[1] + 0.5) * self.cell_size
                    world_y = (guard_pos[0] + 0.5) * self.cell_size
                    radius_world = self.detection_radius * self.cell_size
                    
                    circle = plt.Circle((world_x, world_y), radius_world, 
                                      color='blue', fill=False, linestyle='--', 
                                      linewidth=1.5, alpha=0.3, zorder=5)
                    self.ax.add_patch(circle)
                    self.detection_circles.append(circle)

    def show_no_solution(self):
        """Display no solution message."""
        self.no_solution_text = self.ax.text(0.5, 0.5, 'CBS Timeout', 
                                            transform=self.ax.transAxes, ha='center', va='center',
                                            fontsize=16, color='red', fontweight='bold')

    def show(self):
        """Display the interactive plot."""
        plt.tight_layout()
        plt.show()

def main():
    print("Conflict Based Search - Multi-Agent Pathfinding")
    print("-" * 50)
    
    # Create sample problem with convex obstacles
    grid, agents, obstacles, workspace_width, workspace_height, cell_size = create_sample_problem_with_obstacles()
    
    planner = InteractivePlanner(grid, agents, obstacles, workspace_width, workspace_height, cell_size)
    planner.show()

if __name__ == "__main__":
    main()