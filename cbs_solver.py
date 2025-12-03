import heapq
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from copy import deepcopy

######### CBS SOLVER #########

@dataclass
class Agent:
    id: int
    start: Tuple[float, float]
    goal: Tuple[float, float]

@dataclass
class Constraint:
    # Constraint on agent movement for a specific time step
    agent_id: int
    time: int
    position: Tuple[float, float]

@dataclass
class CBSNode:
    constraints: List[Constraint]
    solution: Dict[int, List[Tuple[int, int]]]
    cost: int
    
    def __lt__(self, other):
        return self.cost < other.cost

class ConflictBasedSearch:

    def __init__(self, grid: List[List[int]], agents: List[Agent]):
        self.grid = grid
        self.agents = agents
        self.rows = len(grid)
        self.cols = len(grid[0])
        
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        # check if in bounds and not an obstacle
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] == 0) # no obstacle
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        # neighboring squares/positions (including staying in place).
        row, col = pos
        neighbors = []
        
        # add current position (if just want to wait)
        neighbors.append((row, col)) # assuming is already valid bc we are here
        # Add up, down, left, right if valid
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        # Manhattan distance
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # does A* search for single agent with constraints
    def low_level_search(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[int, int]]]:

        agent_constraints = [c for c in constraints if c.agent_id == agent.id] # only grab constraints for this agent
        
        # add the vertex constraints into a set for faster lookup
        vertex_constraints = set()
        for c in agent_constraints:
            vertex_constraints.add((c.time, c.position))
        
        open_list = []
        heapq.heappush(open_list, (0, 0, agent.start, [agent.start]))
        closed_set = set()
        
        while open_list:
            f_cost, g_cost, current_pos, path = heapq.heappop(open_list)
            
            if current_pos == agent.goal: # done if at goal
                return path
            
            state = (g_cost, current_pos)
            if state in closed_set: # skip if already closed
                continue
            closed_set.add(state)
            
            # check neighbors
            for next_pos in self.get_neighbors(current_pos):
                new_g_cost = g_cost + 1 # each move is cost 1
                
                if (new_g_cost, next_pos) in vertex_constraints: # skip if not allowed
                    continue
                
                new_f_cost = new_g_cost + self.heuristic(next_pos, agent.goal)
                new_path = path + [next_pos]
                
                heapq.heappush(open_list, (new_f_cost, new_g_cost, next_pos, new_path))
        
        return None  # no path found
    
    # find all the conflicts in a solution. returns list of (agent1_id, agent2_id, time, position)
    def detect_conflicts(self, solution: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, int, Tuple[int, int]]]:
        conflicts = []
        max_length = max(len(path) for path in solution.values())
        
        # Check each time step to find overlap
        for t in range(max_length):
            agents_at_t = {}
            
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t]
                else:
                    pos = path[-1] # agent stays at goal after reaching it
                
                if pos in agents_at_t:
                    other_agent = agents_at_t[pos] # found conflict, add it to list
                    conflicts.append((agent_id, other_agent, t, pos))
                else:
                    agents_at_t[pos] = agent_id
        
        return conflicts
    
    # this is the main CBS solver function, but I changed it to accept previous solution for optimization and try to reuse paths
    def solve(self, previous_solution: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        import time
        start_time = time.time()
        timeout = 5.0  # seconds

        plan_low_level = True
        # If we have a previous solution, try to reuse paths intelligently
        if previous_solution is not None:
            optimized_solution = self._try_optimize_paths(previous_solution)
            if optimized_solution is not None:
                print("Successfully optimized paths from previous solution")
                plan_low_level = False # no need to replan low level for all agents

                if not self.detect_conflicts(optimized_solution):
                    return optimized_solution # no conflicts, done. otherwise continue with CBS with the new paths
        
        if plan_low_level:
            print("Falling back to full CBS recompute")
            # init CBS
            root = CBSNode(constraints=[], solution={}, cost=0)
            for agent in self.agents:
                path = self.low_level_search(agent, [])
                if path is None:
                    return None
                root.solution[agent.id] = path
        else:
            root = CBSNode(constraints=[], solution=optimized_solution, cost=0) # init w optimized solution
        
        root.cost = sum(len(path) for path in root.solution.values())
        open_list = [root]
        heapq.heapify(open_list)
        
        while open_list:
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"CBS timeout after {timeout}s")
                return None

            current = heapq.heappop(open_list)
            conflicts = self.detect_conflicts(current.solution)
            
            if not conflicts:
                return current.solution # good to go
            
            conflict = conflicts[0] # grab first conflict
            agent1_id, agent2_id, conflict_time, position = conflict
            
            for constrained_agent in [agent1_id, agent2_id]: # want to branch on both agents
                child = CBSNode(
                    constraints=deepcopy(current.constraints),
                    solution=deepcopy(current.solution),
                    cost=0
                )

                constraint = Constraint(constrained_agent, conflict_time, position)
                child.constraints.append(constraint)
                
                constrained_agent_obj = next(a for a in self.agents if a.id == constrained_agent)
                new_path = self.low_level_search(constrained_agent_obj, child.constraints) # replan with new constraint
                
                if new_path is not None:
                    child.solution[constrained_agent] = new_path
                    child.cost = sum(len(path) for path in child.solution.values())
                    heapq.heappush(open_list, child)
        
        return None

    # try to reuse previous paths and just recompute the end parts
    def _try_optimize_paths(self, previous_solution: Dict[int, List[Tuple[int, int]]]) -> Optional[Dict[int, List[Tuple[int, int]]]]:

        optimized = {}
        
        for agent in self.agents:
            old_path = previous_solution[agent.id]

            # First, find where to clip the front based on new start position bc the agent keeps moving forward along old path
            start_clip_index = self._find_path_clip_point(agent.start, old_path)
            if start_clip_index is None:
                print("New start not on old path")
                return None  # New start not on old path
            # Clip the front of the path
            old_path = old_path[start_clip_index:]

            # grab old start and goal
            old_start = old_path[0]
            old_goal = old_path[-1]
            
            # If goal never changed, reuse entire path once clipping the start
            if agent.goal == old_goal:
                optimized[agent.id] = old_path
                continue

            # check if obstacle crossing sides changed. If it did, we need full recompute because path topology might change
            if self._obstacle_crossing_changed(old_start, old_goal, agent.start, agent.goal):
                print("Obstacle crossing side changed for agent", agent.id)
                return None  # Full recompute needed
            
            # partial recompute from radius R around new goal
            recompute_path = self._partial_recompute(agent, old_path, 5) # search radius 5, even though only move 4 at a time
            if recompute_path is None:
                return None  # Fall back
            
            # Cleaning up path, check for opposing directions
            cleaned_path = self._remove_opposing_directions(recompute_path, agent)
            if cleaned_path is None:
                return None  # Fall back
            optimized[agent.id] = cleaned_path
        
        # return optimized solution for initialization of CBS
        return optimized

    # Get obstacle centers and check if the line from start to goal crosses different sides of any obstacle
    def _obstacle_crossing_changed(self, old_start: Tuple[int, int], old_goal: Tuple[int, int],
                                    new_start: Tuple[int, int], new_goal: Tuple[int, int]) -> bool:
        old_crossings = self._get_obstacle_crossings(old_start, old_goal)
        new_crossings = self._get_obstacle_crossings(new_start, new_goal)
        
        # If any obstacle crossing side changed, need full recompute
        for obs_idx in old_crossings:
            if obs_idx in new_crossings and old_crossings[obs_idx] != new_crossings[obs_idx]:
                return True
        
        return False

    # this function finds all obstacles crossed by line from start to goal, and which side (left/right)
    def _get_obstacle_crossings(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[Tuple[int,int], str]:
        crossings = {}
        
        # Find all obstacle cells and their approximate centers
        obstacle_clusters = self._find_obstacle_clusters()
        
        for center, cells in obstacle_clusters.items():
            side = self._line_crosses_obstacle_side(start, goal, center)
            if side:
                crossings[center] = side
        
        return crossings

    # gets the obstacle center of mass
    def _find_obstacle_clusters(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        visited = set()
        clusters = {}
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1 and (r, c) not in visited:
                    # breadth-first search to find connected obstacle with only the grid cells
                    cluster = []
                    queue = [(r, c)]
                    visited.add((r, c))
                    
                    while queue:
                        cr, cc = queue.pop(0)
                        cluster.append((cr, cc))
                        
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < self.rows and 0 <= nc < self.cols and
                                self.grid[nr][nc] == 1 and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    
                    # calc center of mass
                    avg_r = sum(pos[0] for pos in cluster) / len(cluster)
                    avg_c = sum(pos[1] for pos in cluster) / len(cluster)
                    center = (int(avg_r), int(avg_c))
                    clusters[center] = cluster
        
        return clusters

    # check the side that the line from start to goal crosses the obstacle center
    def _line_crosses_obstacle_side(self, start: Tuple[int, int], goal: Tuple[int, int],
                                    obstacle_center: Tuple[int, int]) -> Optional[str]:

        # Line vector from start to goal
        dx = goal[1] - start[1]
        dy = goal[0] - start[0]
        
        # Vector from start to obstacle center
        ocx = obstacle_center[1] - start[1]
        ocy = obstacle_center[0] - start[0]
        
        # Cross product shows what side the path is on
        cross = dx * ocy - dy * ocx
        
        return 'left' if cross > 0 else 'right'

    # recompute the part of path within radius R of new goal
    def _partial_recompute(self, agent: Agent, old_path: List[Tuple[int, int]], radius: int) -> Optional[List[Tuple[int, int]]]:
        goal_r, goal_c = agent.goal
        
        # Find where old path enters the radius around new goal
        split_index = None
        for i, pos in enumerate(old_path):
            dist = abs(pos[0] - goal_r) + abs(pos[1] - goal_c)  # Manhattan bc agent moves that way
            if dist <= radius:
                split_index = i
                break
        
        if split_index is None:
            print("Goal too far from old path:", agent.goal, "old path end:", old_path[-1])
            return None  # Goal too far from old path, full recompute needed
        
        # Keep prefix of clipped path, recompute suffix
        prefix = old_path[:split_index]
        
        # to find end of path, update agent start to split point and do low level search
        original_start = agent.start
        agent.start = prefix[-1] if prefix else agent.start
        suffix = self.low_level_search(agent, [])
        agent.start = original_start  # Restore
        
        if suffix is None:
            print("Low-level search failed")
            return None
        
        final_path = prefix[:-1] + suffix  # don't want to double count split point

        return final_path

    def _find_path_clip_point(self, new_start: Tuple[int, int], old_path: List[Tuple[int, int]]) -> Optional[int]:
        # find match point on old path equal to new start
        
        for i, pos in enumerate(old_path):
            if pos == new_start:
                return i  # Exact match
        
        return None  # No match found

    # find opposing directions in path and recompute from there
    def _remove_opposing_directions(self, path: List[Tuple[int, int]], agent: Agent) -> List[Tuple[int, int]]:
        if len(path) < 3:
            return path # too short to have opposing directions
        
        # build direction sequence
        directions = []
        for i in range(len(path) - 1):
            dr = path[i+1][0] - path[i][0]
            dc = path[i+1][1] - path[i][1]
            if dr != 0 or dc != 0:
                directions.append((dr, dc, i))
        
        # get earliest opposing pair
        earliest_conflict = None
        for i in range(len(directions) - 1):
            for j in range(i + 1, len(directions)):
                dr1, dc1, idx1 = directions[i]
                dr2, dc2, idx2 = directions[j]
                
                # Check if opposing
                if (dr1 == -dr2 and dr1 != 0) or (dc1 == -dc2 and dc1 != 0):
                    if earliest_conflict is None or idx1 < earliest_conflict:
                        earliest_conflict = idx1
                    break
        
        if earliest_conflict is not None:
            print(f"Found opposing directions at index {earliest_conflict}")
            
            # temporary agent starting at conflict point and recompute it
            temp_agent = Agent(
                id=agent.id,
                start=path[earliest_conflict],
                goal=agent.goal
            )
            suffix = self.low_level_search(temp_agent, [])
            
            if suffix is None:
                print("Failed to recompute from opposing direction conflict")
                return None
            
            # Combine: keep path up to conflict, add recomputed suffix
            return path[:earliest_conflict] + suffix
        
        return path