# Benchmark functions (non-visual) reusing existing classes/functions
from copy import deepcopy as _deepcopy
import random, time
from main import ConflictBasedSearch
from main import create_sample_problem_with_obstacles

def benchmark_cbs(grid, agents, num_moves=100, goal_step=2):

    working_agents = _deepcopy(agents)
    # Build initial solution
    cbs = ConflictBasedSearch(grid, working_agents)
    solution = cbs.solve()
    if solution is None:
        return {'moves_executed':0,'avg_time':None,'min_time':None,'max_time':None,'all_times':[]}
    
    current_positions = {aid: path[0] for aid, path in solution.items()}
    timestep = 0
    times = []
    moves_executed = 0
    directions = ['up','down','left','right']
    
    initial_snapshot = _deepcopy(working_agents)
    while moves_executed < num_moves:
        # Advance simulated time one step along paths
        max_len = max(len(p) for p in solution.values())
        if timestep + 1 < max_len:
            timestep += 1
            for aid, path in solution.items():
                current_positions[aid] = path[timestep] if timestep < len(path) else path[-1]
        # Restart if any agent at its goal
        if any(current_positions[a.id] == a.goal for a in working_agents):
            working_agents = _deepcopy(initial_snapshot)
            cbs = ConflictBasedSearch(grid, working_agents)
            solution = cbs.solve()
            if solution is None:
                break
            current_positions = {aid: path[0] for aid, path in solution.items()}
            timestep = 0
        # Random goal move for agent 1
        a1 = next(a for a in working_agents if a.id == 1)
        gr,gc = a1.goal
        d = random.choice(directions)
        if d == 'up': new_goal = (gr-goal_step, gc)
        elif d == 'down': new_goal = (gr+goal_step, gc)
        elif d == 'left': new_goal = (gr, gc-goal_step)
        else: new_goal = (gr, gc+goal_step)
        if not cbs.is_valid_position(new_goal):
            continue
        # Update starts
        for ag in working_agents:
            ag.start = current_positions[ag.id]
        a1.goal = new_goal
        start = time.perf_counter()
        cbs = ConflictBasedSearch(grid, working_agents)
        solution_new = cbs.solve()
        elapsed = time.perf_counter() - start
        if solution_new is None:
            # revert if unsolvable
            a1.goal = (gr,gc)
            cbs = ConflictBasedSearch(grid, working_agents)
            solution_new = cbs.solve()
            if solution_new is None:
                break
            continue
        solution = solution_new
        times.append(elapsed)
        moves_executed += 1
        timestep = 0
        current_positions = {aid: path[0] for aid, path in solution.items()}
    return {
        'moves_executed': moves_executed,
        'avg_time': (sum(times)/len(times)) if times else None,
        'min_time': min(times) if times else None,
        'max_time': max(times) if times else None,
        'all_times': times
    }

def benchmark_cbs_sample(num_moves=100, goal_step=2):
    """Convenience wrapper: builds sample problem then calls benchmark_cbs."""
    grid, agents, obstacles, w, h, cell_size = create_sample_problem_with_obstacles()
    return benchmark_cbs(grid, agents, num_moves=num_moves, goal_step=goal_step)

def plot_computation_times(stats, title="CBS Recomputation Times"):
    """Create a boxplot of computation times from benchmark results.
    
    Parameters:
        stats (dict): Result from benchmark_cbs or benchmark_cbs_sample
        title (str): Plot title
    """
    import matplotlib.pyplot as plt
    
    times = stats['all_times']
    if not times:
        print("No timing data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([times], labels=['Recomputation Time'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = f"n={len(times)}\nμ={stats['avg_time']:.4f}s\nmin={stats['min_time']:.4f}s\nmax={stats['max_time']:.4f}s"
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    stats = benchmark_cbs_sample(num_moves=100, goal_step=2)
    print(f"Executed {stats['moves_executed']} moves")
    print(f"Average time: {stats['avg_time']:.6f}s")
    print(f"Min: {stats['min_time']:.6f}s, Max: {stats['max_time']:.6f}s")

    plot_computation_times(stats)