import heapq
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Agent:
    id: int
    start: Tuple[float, float]
    goal: Tuple[float, float]

@dataclass
class Obstacle:
    vertices: List[Tuple[float, float]]

def point_in_convex_polygon(point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> bool:
    # Check if a point is inside a convex polygon using the cross product method. Assumes vertices are ordered
    x, y = point
    n = len(vertices)
    
    # Check if point is on the same side of all edges
    sign = None
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        
        # Cross product of edge vector and point vector
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        
        if abs(cross_product) < 1e-10:  # Point is on the edge
            continue
            
        if sign is None:
            sign = cross_product > 0
        elif (cross_product > 0) != sign:
            return False
    
    return True

def create_grid_from_obstacles(workspace_width: float, workspace_height: float, 
                              resolution: float, obstacles: List[Obstacle]) -> Tuple[List[List[int]], float]:
    # makes grid of 1s and 0s representing obstacles from convex polygons
 
    # Calculate grid dimensions based on workspace size and resolution
    grid_width = int(np.ceil(workspace_width / resolution))
    grid_height = int(np.ceil(workspace_height / resolution))
    
    # Calculate actual cell size
    actual_cell_size_x = workspace_width / grid_width
    actual_cell_size_y = workspace_height / grid_height
    
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    
    for row in range(grid_height):
        for col in range(grid_width):
            # Calculate cell center in world coordinates
            cell_center_x = (col + 0.5) * actual_cell_size_x
            cell_center_y = (row + 0.5) * actual_cell_size_y
            
            # Check if cell center is inside any obstacle
            for obstacle in obstacles:
                if point_in_convex_polygon((cell_center_x, cell_center_y), obstacle.vertices):
                    grid[row][col] = 1
                    break

    return grid, actual_cell_size_x

def create_sample_problem_with_obstacles():
    # Just makes an example problem with convex polygon obstacles

    # Define workspace dimensions (in world coordinates)
    workspace_width = 20.0
    workspace_height = 15.0
    
    # Define discretization resolution (cell size in world coordinates)
    resolution = 0.25  # Each grid cell represents X x X world units
    
    # Makes some obstacles as convex polygons (in world coordinates)
    obstacles = [
        # Rectangular obstacle in the upper left
        Obstacle(vertices=[(2.0, 0), (6.0, 0), (6.0, 4.0), (2.0, 4.0)]),
        
        # Triangular obstacle in the middle
        Obstacle(vertices=[(8.0, 6.0), (12.0, 0.5), (10.0, 12.0)]),
        
        # Another rectangular obstacle
        Obstacle(vertices=[(15.0, 2.0), (18.0, 2.0), (18.0, 5.0), (15.0, 5.0)]),
        
        # Small triangular obstacle
        Obstacle(vertices=[(4.0, 9.0), (7.0, 8.0), (5.0, 11.5)])
    ]
    
    # Create grid from obstacles
    grid, cell_size = create_grid_from_obstacles(workspace_width, workspace_height, 
                                                resolution, obstacles)
    
    # Convert world coordinates to grid coordinates
    def world_to_grid(world_pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = world_pos
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        return (grid_y, grid_x)  # Note: (row, col) format for grid
    
    # Create agents (positions in grid coordinates)
    agents = [
        #Agent(id=0, start=world_to_grid((1.0, 1.0)), goal=world_to_grid((18.0, 13.0))),  # Agent 0
        Agent(id=1, start=world_to_grid((18.0, 1.0)), goal=world_to_grid((1.0, 13.0))),   # Agent 1
        Agent(id=2, start=world_to_grid((15.0, 1.0)), goal=world_to_grid((1.0, 1.0)))   # Agent 2
    ]
    
    return grid, agents, obstacles, workspace_width, workspace_height, cell_size

@dataclass
class Constraint:
    # Constraint on agent movement
    agent_id: int
    time: int
    position: Tuple[float, float]
    constraint_type: str = "vertex"  # "vertex" or "edge"

@dataclass
class CBSNode:
    # Node in the CBS high-level search tree.
    constraints: List[Constraint]
    solution: Dict[int, List[Tuple[int, int]]]
    cost: int
    
    def __lt__(self, other):
        return self.cost < other.cost

class ConflictBasedSearch:
    # Conflict Based Search algorithm for multi-agent pathfinding.

    def __init__(self, grid: List[List[int]], agents: List[Agent]):
        self.grid = grid
        self.agents = agents
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        # Check if a position is valid (within bounds and not an obstacle).
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] == 0)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Get valid neighboring positions (including staying in place).
        row, col = pos
        neighbors = []
        
        # Add current position (if just want to wait)
        neighbors.append((row, col))
        
        # Add adjacent positions (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        # Manhattan distance heuristic.
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def low_level_search(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[int, int]]]:
        """
        A* search for a single agent considering constraints.
        Returns the path or None if no path exists.
        """
        # Filter constraints for this agent
        agent_constraints = [c for c in constraints if c.agent_id == agent.id]
        
        # Create constraint lookup for efficiency
        vertex_constraints = set()
        for c in agent_constraints:
            if c.constraint_type == "vertex":
                vertex_constraints.add((c.time, c.position))
        
        # A* search
        open_list = []
        heapq.heappush(open_list, (0, 0, agent.start, [agent.start]))
        closed_set = set()
        
        while open_list:
            f_cost, g_cost, current_pos, path = heapq.heappop(open_list)
            
            # Check if we reached the goal
            if current_pos == agent.goal:
                return path
            
            state = (g_cost, current_pos)
            if state in closed_set:
                continue
            closed_set.add(state)
            
            # Explore neighbors
            for next_pos in self.get_neighbors(current_pos):
                new_g_cost = g_cost + 1
                
                # Check constraints
                if (new_g_cost, next_pos) in vertex_constraints:
                    continue
                
                new_f_cost = new_g_cost + self.heuristic(next_pos, agent.goal)
                new_path = path + [next_pos]
                
                heapq.heappush(open_list, (new_f_cost, new_g_cost, next_pos, new_path))
        
        return None  # No path found
    
    def detect_conflicts(self, solution: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, int, Tuple[int, int]]]:
        # Find list of all vertex conflicts in a given solution: (agent1_id, agent2_id, time, position)
        conflicts = []
        
        # Find maximum path length
        max_length = max(len(path) for path in solution.values())
        
        # Check for vertex conflicts at each time step
        for t in range(max_length):
            positions_at_t = {}
            
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t]
                else:
                    # Agent stays at goal after reaching it
                    pos = path[-1]
                
                if pos in positions_at_t:
                    # Conflict detected
                    other_agent = positions_at_t[pos]
                    conflicts.append((agent_id, other_agent, t, pos))
                else:
                    positions_at_t[pos] = agent_id
        
        return conflicts
    
    def solve(self) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        # MAIN CBS ALGORITHM

        # init root node
        root = CBSNode(constraints=[], solution={}, cost=0)
        
        # Find initial solution without constraints
        for agent in self.agents:
            path = self.low_level_search(agent, [])
            if path is None:
                return None  # No solution possible
            root.solution[agent.id] = path
        
        # Calculate initial cost (sum of path lengths)
        root.cost = sum(len(path) for path in root.solution.values())
        
        # Initialize open list
        open_list = [root]
        heapq.heapify(open_list)
        
        while open_list:
            current = heapq.heappop(open_list)
            
            # Detect conflicts
            conflicts = self.detect_conflicts(current.solution)
            
            if not conflicts:
                # No conflicts, solution found
                return current.solution
            
            # Take the first conflict and create child nodes
            conflict = conflicts[0]
            agent1_id, agent2_id, time, position = conflict
            
            # Create two child nodes with constraints
            for constrained_agent in [agent1_id, agent2_id]:
                child = CBSNode(
                    constraints=deepcopy(current.constraints),
                    solution=deepcopy(current.solution),
                    cost=0
                )
                
                # Add constraint
                constraint = Constraint(constrained_agent, time, position)
                child.constraints.append(constraint)
                
                # Replan for the constrained agent
                constrained_agent_obj = next(a for a in self.agents if a.id == constrained_agent)
                new_path = self.low_level_search(constrained_agent_obj, child.constraints)
                
                if new_path is not None:
                    child.solution[constrained_agent] = new_path
                    child.cost = sum(len(path) for path in child.solution.values())
                    heapq.heappush(open_list, child)
        
        return None  # No solution found

def plot_solution(solution: Dict[int, List[Tuple[int, int]]], grid: List[List[int]], 
                 obstacles: List[Obstacle] = None, workspace_width: float = None, 
                 workspace_height: float = None, cell_size: float = 1.0):

    # Create a matplotlib visualization of the solution with time-colored paths.
    if solution is None:
        print("No solution to plot!")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot original obstacle polygons if provided
    if obstacles:
        for i, obstacle in enumerate(obstacles):
            vertices = np.array(obstacle.vertices + [obstacle.vertices[0]])  # Close the polygon
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2, alpha=0.7)
            ax.fill(vertices[:, 0], vertices[:, 1], color='gray', alpha=0.3)
    
    # Plot grid obstacles (for reference)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 1:  # Obstacle
                # Convert grid coordinates to world coordinates
                world_x = j * cell_size
                world_y = i * cell_size
                ax.add_patch(plt.Rectangle((world_x, world_y), cell_size, cell_size, 
                                         facecolor='red', alpha=0.4, edgecolor='red'))
    
    # Define colors for agents
    agent_colors = ['blue', 'red', 'green', 'purple', 'orange']
    agent_offsets = [(-0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (0.02, -0.02), (0, 0)] # so they don't overlap in visual
    
    # Plot paths for each agent
    max_time = max(len(path) for path in solution.values())
    
    for agent_id, path in solution.items():
        agent_color = agent_colors[agent_id % len(agent_colors)]
        
        # Extract x and y coordinates (convert grid coordinates to world coordinates)
        x_coords = [(pos[1] + 0.5) * cell_size + agent_offsets[agent_id][0] for pos in path]  # Column to world x
        y_coords = [(pos[0] + 0.5) * cell_size + agent_offsets[agent_id][1] for pos in path]  # Row to world y

        # Create color map for time progression
        times = np.arange(len(path))
        colors = plt.cm.plasma(times / max(times)) if len(times) > 1 else [agent_color]
        
        # Plot each timestep as individual dots with changing colors
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            # Use different markers for start, intermediate, and goal positions
            if i == 0:  # Start position
                marker = 's'  # Square
                size = 200
                edgecolor = 'black'
                linewidth = 3
                label = f'Agent {agent_id} Start' if agent_id == 0 or agent_id == 1 else None
            elif i == len(path) - 1:  # Goal position
                marker = 's'  # another square
                size = 200
                edgecolor = 'black'
                linewidth = 3
                label = f'Agent {agent_id} Goal' if agent_id == 0 or agent_id == 1 else None
            else:  # Intermediate positions
                marker = 'o'  # Circle
                size = 50
                edgecolor = 'white'
                linewidth = 1
                label = None
            
            # Plot the dot with time-based color
            scatter = ax.scatter(x, y, c=[colors[i]], s=size, marker=marker,
                               edgecolors=edgecolor, linewidth=linewidth,
                               label=label, zorder=5, alpha=0.8)
            
            # Add timestep number on each dot
            ax.annotate(str(i), (x, y), 
                       ha='center', va='center',
                       fontsize=8, fontweight='bold',
                       color='white' if marker == 'o' else 'black')
    
    # Set up the plot
    if workspace_width and workspace_height:
        ax.set_xlim(0, workspace_width)
        ax.set_ylim(0, workspace_height)
        ax.set_xlabel('X (world coordinates)')
        ax.set_ylabel('Y (world coordinates)')
        ax.set_title(f'Multi-Agent Pathfinding Solution\n(Workspace: {workspace_width}x{workspace_height}, Resolution: {cell_size:.2f})')
    else:
        ax.set_xlim(-0.5, len(grid[0]) - 0.5)
        ax.set_ylim(-0.5, len(grid) - 0.5)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Multi-Agent Pathfinding Solution')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis to match grid convention (0,0 at top-left)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

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
        
        # Set up the figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initialize containers for dynamic elements
        self.agent_current_positions = {}  # Current positions of agents
        self.agent_position_objects = {}   # Visual objects for current positions
        self.no_solution_text = None

        # Draw static background once
        self.setup_static_background()
        
        # Calculate and plot initial solution
        self.calculate_new_solution()
        
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
        if (0 <= row < len(self.grid) and 
            0 <= col < len(self.grid[0]) and 
            self.grid[row][col] == 0):
            return True
        return False
    
    def on_key_press(self, event):
        """Handle keyboard input to move agent 1's goal."""
        if self.game_over:
            return
        
        if event.key not in ['up', 'down', 'left', 'right']:
            return
            
        # Get current goal of agent 1
        agent1 = next(a for a in self.agents if a.id == 1)
        current_goal = agent1.goal
        
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
        
        # Check if new position is valid
        if self.is_valid_goal(new_goal):
            agent1.goal = new_goal
            self.draw_goal_marker()
            self.calculate_new_solution()
        else:
            print(f"Invalid goal position: {new_goal}")
    
    def calculate_new_solution(self):
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
        solution = self.cbs.solve()
        
        calc_time = time.time() - start_time
        print(f"New path calculated in {calc_time:.20f}s")
        
        # Stop any existing animation
        if self.animation_timer:
            self.animation_timer.event_source.stop()
        
        # Clear dynamic elements
        self.clear_dynamic_elements()
        
        if solution is None:
            self.show_no_solution()
            self.game_over = True
        else:
            self.current_solution = solution
            self.current_timestep = 0
            self.game_over = False
            
            # Initialize agent positions to start positions (which are now current positions)
            for agent_id, path in solution.items():
                self.agent_current_positions[agent_id] = path[0]
            
            # Start the animation
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
                                        interval=100, repeat=True, blit=False)

    def update_agent_positions(self, frame):
        """Move agents one step forward in their paths."""
        if self.game_over or not self.current_solution:
            return
        
        self.current_timestep += 1
        
        # Update each agent's position
        all_at_goal = True
        for agent_id, path in self.current_solution.items():
            if self.current_timestep < len(path):
                self.agent_current_positions[agent_id] = path[self.current_timestep]
                all_at_goal = False
            else:
                # Agent has reached the end of path, stay at goal
                self.agent_current_positions[agent_id] = path[-1]
        
        # Update the visual representation
        self.update_agent_visuals()
        self.draw_goal_marker()
        
        # Check win/lose conditions
        if all_at_goal:
            self.end_game("GAME OVER - Agent reached the goal! You lost!")

    def draw_goal_marker(self):
        """Draw the goal position that the user is controlling."""
        # Get Agent 1's goal position
        agent1 = next(a for a in self.agents if a.id == 1)
        goal_pos = agent1.goal
        
        # Convert to world coordinates
        world_x = (goal_pos[1] + 0.5) * self.cell_size
        world_y = (goal_pos[0] + 0.5) * self.cell_size
        
        # Remove existing goal marker if it exists
        if hasattr(self, 'goal_marker') and self.goal_marker:
            self.goal_marker.remove()
        
        # Draw goal as a large square with distinctive appearance
        self.goal_marker = self.ax.scatter(world_x, world_y, c='gold', s=600, 
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
            if agent_id == 1:
                # Special large circle for agent 1 (the one we control)
                marker, size, edgecolor = 'o', 400, 'black'
            else:
                # Square for other agents
                marker, size, edgecolor = 's', 300, 'black'
            
            pos_obj = self.ax.scatter(world_x, world_y, c=agent_color, s=size, 
                                    marker=marker, edgecolors=edgecolor, linewidth=3,
                                    zorder=10, alpha=0.9)
            
            self.agent_position_objects[agent_id] = pos_obj
        
        # Update display
        self.fig.canvas.draw_idle()

    def show_no_solution(self):
        """Display no solution message."""
        self.no_solution_text = self.ax.text(0.5, 0.5, 'No Solution Found!', 
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
    # grid, agents, obstacles, workspace_width, workspace_height, cell_size = create_sample_problem_with_obstacles()
    
    # planner = InteractivePlanner(grid, agents, obstacles, workspace_width, workspace_height, cell_size)
    # planner.show()

if __name__ == "__main__":
    main()