# 导入部分添加
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from textwrap import wrap
import time
from gcaa.algorithms.greedy import GCAASolution
from gcaa.tools.plotting import plotMapAllocation
from gcaa.tools.serialize import make_json_serializable
from gcaa.core.control import OptimalControlSolution


def optimal_control_dta(
        na: int = 10, nt: int = 10, 
        map_width: int = 10, 
        CommLimit: bool = False, 
        comm_distance: float = 3.0,
        v_a_max: float = 1.0,
        v_t_max: float = 0.1,
        r_a: float = 0.1, r_t: float = 0.1,
        n_rounds: int = 10,
        use_GCAA: bool = True,
        n_rounds_loop: int = 10,
        seed: int = 1234,
        debug: bool = False,
        animate: bool = True,
        save: bool = False,
        save_dir: str = "",
        filename: str = "",
        verbose: bool = False
):
    """
    Dynamic Task Agent Allocation (DTA) with Optimal Control implementation
    
    Parameters
    ----------
    na: int
        Number of agents
    nt: int
        Number of tasks
    map_width: int
        Width of the map (square map)
    CommLimit: bool
        Whether to enforce communication limits between agents
    comm_distance: float
        Maximum communication distance between agents
    v_a_max: float
        Maximum agent velocity
    v_t_max: float
        Maximum task velocity
    r_a: float
        Agent radius
    r_t: float
        Task radius
    n_rounds: int
        Number of simulation rounds
    use_GCAA: bool
        Whether to use GCAA algorithm for task allocation
    n_rounds_loop: int
        Number of rounds in each loop
    seed: int
        Random seed
    debug: bool
        Debug mode
    animate: bool
        Whether to animate the simulation
    save: bool
        Whether to save the animation
    save_dir: str
        Directory to save the animation
    filename: str
        Filename for the saved animation
    verbose: bool
        Verbose mode
    
    Returns
    -------
    dict
        Dictionary containing simulation results
    """
    np.random.seed(seed)
    
    # Initialize agents and tasks positions
    pos_a = np.random.rand(na, 2) * map_width  # Agents position
    v_a = np.random.rand(na, 2) * v_a_max  # Agents velocity
    pos_t = np.random.rand(nt, 2) * map_width  # Tasks position
    v_t = np.random.rand(nt, 2) * v_t_max  # Tasks velocity
    
    # Create agents and tasks objects
    class Agents:
        def __init__(self, na, map_width):
            # Required by GCAASolution
            self.N = na
            self.Pos = np.random.rand(na, 2) * map_width
            self.v_a = np.random.rand(na, 2) * v_a_max
            self.Lt = np.zeros(na, dtype=int)
            self.previous_task = -np.ones(na, dtype=int)
            self.previous_winnerBids = np.zeros(na)
            self.kdrag = 0.0  # Default value, adjust as needed
            self.rin_task = np.zeros((na, 2))
            self.vin_task = np.zeros((na, 2))
            
            # Additional properties used in the simulation
            self.r_a = r_a
            self.v_a_max = v_a_max
            self.rout_task = np.zeros((na, 2))
            self.los = np.ones(na, dtype=bool)
            self.J = np.zeros(na)
            self.U = np.zeros(na)
            self.rt = np.zeros(na)
    
    class Tasks:
        def __init__(self, nt, map_width):
            # Required by GCAASolution
            self.N = nt
            self.Pos = np.random.rand(nt, 2) * map_width
            self.tf = np.zeros(nt)  # Task finish times
            self.r_bar = np.zeros(nt)  # Default values, adjust as needed
            self.Speed = np.random.rand(nt, 2) * v_t_max
            self.task_type = np.zeros(nt, dtype=int)  # 0 for default type
            self.radius = r_t * np.ones(nt)
            self.tloiter = np.zeros(nt)  # Loitering time
            self.prob_a_t = np.ones((na, nt))  # Probability of agent-task assignment
            self.lambda_ = 1.0  # Default value, adjust as needed
            
            # Additional properties used in the simulation
            self.v = np.random.rand(nt, 2) * v_t_max
            self.r_t = r_t
            self.S = np.zeros(nt)
            self.rt = np.zeros(nt)
            self.U = np.zeros(nt)
            self.counter = np.zeros(nt, dtype=int)
    
    agents = Agents(na, map_width)
    agents.Pos = pos_a.copy()
    agents.v_a = v_a.copy()
    
    tasks = Tasks(nt, map_width)
    tasks.Pos = pos_t.copy()
    tasks.v = v_t.copy()
    tasks.Speed = v_t.copy()  # Map velocity to Speed attribute expected by GCAASolution
    
    # Initialize simulation variables
    pos_a_loop = pos_a.copy()
    v_a_loop = v_a.copy()
    
    U_next_tot = np.zeros(n_rounds)
    U_tot = np.zeros(n_rounds)
    U_completed_tot = 0.0
    
    completed_tasks_round = []
    completed_tasks = []
    total_completed_tasks = 0
    rt_completed = 0.0

    # 初始化智能体颜色
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 如果颜色数量少于智能体数量，循环使用颜色
    while len(colors) < na:
        colors.extend(colors)
    colors = colors[:na]
    
    # Preallocate lists (MATLAB cell arrays)
    X_full_simu = [None] * n_rounds
    p_GCAA_full_simu = [None] * n_rounds
    
    S_GCAA_ALL_full_simu = np.zeros((n_rounds, nt))
    rt_full_simu = np.zeros((n_rounds, nt))
    
    J = np.zeros((n_rounds, na))
    J_to_completion_target = np.zeros((n_rounds, na))
    
    # cost/reward/utility arrays reused each round
    costs = np.zeros((na, nt))
    rewards = np.zeros((na, nt))
    utility = np.zeros((na, nt))
    
    # Fully connected graph initially (no self links)
    G = ~np.eye(na, dtype=bool)
    
    # Initialize communication overhead
    comm_overhead = 0
    comm_overhead_per_round = np.zeros(n_rounds)
    
    fig, ax = plt.subplots()
    
    def wrap_title(event=None):
        # Width of the figure in pixels
        fig_width_px = fig.get_figwidth() * fig.dpi
        
        # Pick characters-per-line empirically.
        # You can tune the scaling factor if needed.
        max_chars = int(fig_width_px / 7)
        
        wrapped = "\n".join(wrap(title_text, max_chars))
        title.set_text(wrapped)
        fig.canvas.draw_idle()
    
    historical_path = {}
    
    for i_round in range(n_rounds):
        # Update task positions
        pos_t += v_t
        pos_t = np.mod(pos_t, map_width)  # Wrap around map
        
        # Update tasks.Pos for GCAASolution
        tasks.Pos = pos_t.copy()
        
        # Plotting setup
        ax.clear()
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_width)
        
        # Draw communication range if CommLimit is active
        if CommLimit and animate:
            for i in range(na):
                circle = Circle(pos_a_loop[i, :], comm_distance, 
                               fill=False, color='gray', alpha=0.3)
                ax.add_patch(circle)
        
        # Plot agent positions
        ax.plot(pos_a_loop[:, 0], pos_a_loop[:, 1], 'k*', markersize=8, label='Agents')
        
        # Plot task positions
        ax.plot(pos_t[:, 0], pos_t[:, 1], 'rs', markersize=6, label='Tasks')
        
        title_text = f"Round {i_round + 1}/{n_rounds} - Agents: {na}, Tasks: {nt}, CommLimit: {CommLimit}"
        title = ax.set_title(title_text)
        wrap_title()
        
        # Communication graph update if CommLimit active
        if CommLimit:
            # Calculate communication overhead (number of connections)
            round_comm_overhead = 0
            for i in range(na):
                for j in range(i + 1, na):
                    connected = np.linalg.norm(
                        pos_a_loop[i, :] - pos_a_loop[j, :]) < comm_distance
                    G[i, j] = connected
                    G[j, i] = connected
                    if connected:
                        round_comm_overhead += 1
            
            comm_overhead += round_comm_overhead
            comm_overhead_per_round[i_round] = round_comm_overhead

        # Solve allocation with chosen method(s)
        if use_GCAA:
            t0 = time.perf_counter()
            S_GCAA, p_GCAA, S_GCAA_ALL, rt_curr, agents = GCAASolution(
                agents, G, tasks, map_width)
            rt_full_simu[i_round, :] = rt_curr
            t1 = time.perf_counter()

            # Fix: Check if path is empty before accessing p[0]
            alloc_lines = []
            for i, p in enumerate(p_GCAA):
                if p:  # Check if path is not empty
                    alloc_lines.append(f'Agent {i} -> Task {p[0]}')
                else:
                    alloc_lines.append(f'Agent {i} -> No task assigned')
            alloc_cleaned = '\n'.join(alloc_lines)

            if verbose:
                print(
                    f"GCAA round {i_round + 1} ({t1 - t0:.2f}s)\n"
                    f"{alloc_cleaned}\n"
                    f"--------------------"
                )
        else:
            # test fixed task allocation
            p_GCAA = [[0], [1], [3], [1], [2]][:na]
            S_GCAA = 1
            S_GCAA_ALL = np.zeros(nt)
            rt_curr = np.zeros(nt)
            for i in range(na):
                if i < len(p_GCAA) and len(p_GCAA[i]) > 0:
                    task_index = p_GCAA[i][0]  # first task for agent i
                    agents.rin_task[i, :] = pos_t[task_index, :]
        
        U_next_tot[i_round] = S_GCAA
        U_tot[i_round] = U_next_tot[i_round] + U_completed_tot
        
        # Find the optimal control trajectory for the allocation p_GCAA
        # 假设time_step为0.1（可以根据需要调整）
        time_step = 0.1
        X, completed_tasks_round, J_curr, J_to_completion_target[i_round] = OptimalControlSolution(
            pos_a_loop, v_a_loop, pos_t, v_t, r_t, p_GCAA, agents,
            tasks.tf, tasks.tloiter, time_step, n_rounds_loop, na, agents.kdrag
        )
        
        # Update cost array
        if i_round == 0:
            J[i_round, :] = J_curr
        else:
            J[i_round, :] = J[i_round - 1, :] + J_curr
        
        # plot map allocation
        plotMapAllocation(X, n_rounds_loop, na, colors, "GCAA solution")
        
        # accumulate completed tasks reward-time if any
        round_completed_tasks = 0
        for j in completed_tasks_round:
            rt_completed += rt_curr[j]
            completed_tasks.append(j)
            round_completed_tasks += 1
        
        total_completed_tasks += round_completed_tasks
        
        # reset for next round (as in MATLAB)
        completed_tasks_round = []
        
        # unique legend and draw
        SKIP_LABELS = {"GCAA solution", "Comm Range"}
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = [h for h, l in zip(handles, labels) if
                           l not in SKIP_LABELS]
        filtered_labels = [l for l in labels if l not in SKIP_LABELS]
        ax.legend(filtered_handles, filtered_labels)
        
        if animate:
            plt.draw()
            plt.pause(0.001)
        
        # Update agent positions and velocities from X:
        # MATLAB used: pos_a_loop = X(1:2,:,2)'; v_a_loop = X(3:4,:,2)';
        # Assuming X is a numpy array shaped (4, na, n_horizon)
        pos_a_loop = X[0:2, :, 1].T.copy()
        v_a_loop = X[2:4, :, 1].T.copy()
        
        # Update agents.Pos and agents.v_a for next iteration of GCAASolution
        agents.Pos = pos_a_loop.copy()
        agents.v_a = v_a_loop.copy()
        
        # Store simulation data
        historical_path[i_round] = pos_a_loop
        X_full_simu[i_round] = X
        p_GCAA_full_simu[i_round] = p_GCAA
        S_GCAA_ALL_full_simu[i_round, :] = S_GCAA_ALL
        
    U_tot_final = rt_completed - np.sum(J[-1, :])
    print(f"Total utility: {U_tot_final}")
    print(f"Total completed tasks: {total_completed_tasks}")
    print(f"Total communication overhead: {comm_overhead}")
    
    print("Simulation finished successfully.")
    
    return dict(
        historical_path=historical_path,
        total_utility=U_tot_final,
        total_completed_tasks=total_completed_tasks,
        total_communication_overhead=comm_overhead,
        communication_overhead_per_round=comm_overhead_per_round,
        U_next_tot=U_next_tot,
        U_tot=U_tot,
        completed_tasks=completed_tasks,
        rt_completed=rt_completed,
        J=J,
        J_to_completion_target=J_to_completion_target,
        X_full_simu=X_full_simu,
        p_GCAA_full_simu=p_GCAA_full_simu
    )

# Run the simulation if this script is executed directly
if __name__ == "__main__":
    optimal_control_dta()
