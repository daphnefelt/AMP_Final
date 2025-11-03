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
    """
    Check if a point is inside a convex polygon using the cross product method.
    Assumes vertices are ordered (clockwise or counterclockwise).
    """
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
    """
    Create a grid where cells that intersect with any obstacle are marked as 1.
    
    Args:
        workspace_width: Width of the workspace in world coordinates
        workspace_height: Height of the workspace in world coordinates
        resolution: Size of each grid cell in world coordinates
        obstacles: List of convex obstacles defined by vertices
    
    Returns:
        Tuple of (grid, actual_cell_size) where:
        - grid: 2D grid where 0 = free space, 1 = obstacle
        - actual_cell_size: The actual cell size used (may be slightly adjusted)
    """
    # Calculate grid dimensions based on workspace size and resolution
    grid_width = int(np.ceil(workspace_width / resolution))
    grid_height = int(np.ceil(workspace_height / resolution))
    
    # Calculate actual cell size (may be slightly different due to rounding)
    actual_cell_size_x = workspace_width / grid_width
    actual_cell_size_y = workspace_height / grid_height
    
    print(f"Workspace: {workspace_width} x {workspace_height}")
    print(f"Grid dimensions: {grid_width} x {grid_height}")
    print(f"Requested resolution: {resolution}")
    print(f"Actual cell size: {actual_cell_size_x:.3f} x {actual_cell_size_y:.3f}")
    
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
    """Create a sample problem with convex obstacles defined by vertices."""
    # Define workspace dimensions (in world coordinates)
    workspace_width = 20.0
    workspace_height = 15.0
    
    # Define discretization resolution (cell size in world coordinates)
    resolution = 0.25  # Each grid cell represents X x X world units
    
    # Define obstacles as convex polygons (in world coordinates)
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
    
    # Create agents (positions in grid coordinates)
    # Convert world coordinates to grid coordinates
    def world_to_grid(world_pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = world_pos
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        return (grid_y, grid_x)  # Note: (row, col) format for grid
    
    agents = [
        Agent(id=0, start=world_to_grid((1.0, 1.0)), goal=world_to_grid((18.0, 13.0))),  # Agent 0
        Agent(id=1, start=world_to_grid((18.0, 1.0)), goal=world_to_grid((1.0, 13.0))),   # Agent 1
        Agent(id=2, start=world_to_grid((15.0, 1.0)), goal=world_to_grid((1.0, 12.0)))   # Agent 2
    ]
    
    return grid, agents, obstacles, workspace_width, workspace_height, cell_size

@dataclass
class Constraint:
    """Represents a constraint on an agent's movement."""
    agent_id: int
    time: int
    position: Tuple[float, float]
    constraint_type: str = "vertex"  # "vertex" or "edge"

@dataclass
class CBSNode:
    """Node in the CBS high-level search tree."""
    constraints: List[Constraint]
    solution: Dict[int, List[Tuple[int, int]]]
    cost: int
    
    def __lt__(self, other):
        return self.cost < other.cost

class ConflictBasedSearch:
    """Conflict Based Search algorithm for multi-agent pathfinding."""
    
    def __init__(self, grid: List[List[int]], agents: List[Agent]):
        self.grid = grid
        self.agents = agents
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not an obstacle)."""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] == 0)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (including staying in place)."""
        row, col = pos
        neighbors = []
        
        # Add current position (wait action)
        neighbors.append((row, col))
        
        # Add adjacent positions (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
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
        """
        Detect vertex conflicts in the solution.
        Returns list of conflicts: (agent1_id, agent2_id, time, position)
        """
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
        """
        Main CBS algorithm.
        Returns the solution or None if no solution exists.
        """
        # Initialize root node
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

def create_sample_problem():
    """Create a sample problem with 2 agents."""
    # Create a simple grid (0 = free, 1 = obstacle)
    grid = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Create agents
    agents = [
        Agent(id=0, start=(0, 0), goal=(4, 9)),  # Agent 0: top-left to bottom-right
        Agent(id=1, start=(4, 0), goal=(0, 8))   # Agent 1: bottom-left to top-right
    ]
    
    return grid, agents

def print_solution(solution: Dict[int, List[Tuple[int, int]]], grid: List[List[int]]):
    """Print the solution in a readable format."""
    if solution is None:
        print("No solution found!")
        return
    
    print("Solution found!")
    print("\nPaths:")
    for agent_id, path in solution.items():
        print(f"Agent {agent_id}: {path}")

    print("\n")

def plot_solution(solution: Dict[int, List[Tuple[int, int]]], grid: List[List[int]], 
                 obstacles: List[Obstacle] = None, workspace_width: float = None, 
                 workspace_height: float = None, cell_size: float = 1.0):
    """Create a matplotlib visualization of the solution with time-colored paths."""
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

def main():
    """Main function to run the CBS algorithm."""
    print("Conflict Based Search - Multi-Agent Pathfinding")
    print("=" * 50)
    
    # Create sample problem with convex obstacles
    grid, agents, obstacles, workspace_width, workspace_height, cell_size = create_sample_problem_with_obstacles()
    
    print("Obstacles (defined by vertices in world coordinates):")
    for i, obstacle in enumerate(obstacles):
        print(f"Obstacle {i}: {obstacle.vertices}")
    
    print(f"\nWorkspace: {workspace_width} x {workspace_height} (world units)")
    print(f"Cell size: {cell_size:.3f}")
    print(f"Grid dimensions: {len(grid[0])} x {len(grid)}")
    
    print(f"\nAgents (grid coordinates):")
    for agent in agents:
        print(f"Agent {agent.id}: Start {agent.start} -> Goal {agent.goal}")
    
    # Solve using CBS
    cbs = ConflictBasedSearch(grid, agents)
    solution = cbs.solve()
    
    print("DONE")
    # Print solution
    # print("\n" + "=" * 50)
    # print_solution(solution, grid)
    
    # Plot solution with original obstacles
    plot_solution(solution, grid, obstacles, workspace_width, workspace_height, cell_size)

if __name__ == "__main__":
    main()