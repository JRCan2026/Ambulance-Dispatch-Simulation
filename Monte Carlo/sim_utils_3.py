# sim_utils.py
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import osmnx as ox
import pandas as pd
import networkx as nx
from shapely.geometry import Point
from lifelines import KaplanMeierFitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import copy
import time
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Road network
# ============================================================
G = ox.graph_from_place("Quebec, Quebec, Canada", network_type="drive")

# TRAVEL TIME FUNCTION
G = ox.add_edge_speeds(G, fallback=40)
G = ox.add_edge_travel_times(G)

# ============================================================
# CLASS "AMBULANCE"
# ============================================================
class Ambulance:
    def __init__(self, idx, position, current_node):
        self.idx = idx
        self.position = np.array(position)
        self.current_node = current_node
        self.available = True
        self.busy_until = 0

    def update(self, t):
        if t >= self.busy_until:
            self.available = True

    def assign(self, destination, dest_node, travel_time, t, service_time=10):
        self.available = False
        self.busy_until = t + travel_time + service_time
        self.position = np.array(destination)
        self.current_node = dest_node


# ============================================================
# TRAVEL TIME AND COST FUNCTIONS
# ============================================================
def network_time(a, b, G):
    """
    This function computes de shortest path between two given nodes within the road-network.
    """
    node_a = ox.distance.nearest_nodes(G, a[0], a[1]) if isinstance(a, tuple) else a
    node_b = ox.distance.nearest_nodes(G, b[0], b[1]) if isinstance(b, tuple) else b

    try:
        length = nx.shortest_path_length(G, node_a, node_b, weight="travel_time")
        return length / 60
    except nx.NetworkXNoPath:
        return 9999.0

def compute_cost_matrix(ambulances, queue_calls, G):
    """
    This function computes travel times from between patient i and ambulance j.
    """
    ambulances_list = list(ambulances)
    n_a = len(ambulances_list)
    n_c = len(queue_calls)
    cost_matrix = np.zeros((n_a, n_c))
    
    for i, amb in enumerate(ambulances_list): # <---- AQUI
        for j, call_data in enumerate(queue_calls):
            call_node = call_data['node']
            cost_matrix[i, j] = network_time(amb.current_node, call_node, G)

    return cost_matrix


# ============================================================
# CALLS GENERATOR FUNCTION
# ============================================================
def generate_calls(lmbda, city_limits):
    """
    This function initializes emergency calls' positions randomly within the city boundaries
    """
    n_calls = np.random.poisson(lmbda)
    poly = city_limits.union_all()
    xmin, ymin, xmax, ymax = poly.bounds
    valid_calls = []
    
    while len(valid_calls) < n_calls:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        p = Point(x, y)
        if poly.contains(p):
            valid_calls.append((x, y))
            
    return valid_calls

def generate_call_stream(T, lamda, city_limits, seed):
    rng = np.random.default_rng(seed)

    poly = city_limits.union_all()
    xmin, ymin, xmax, ymax = poly.bounds

    call_stream = []

    for t in range(T):
        n_calls = rng.poisson(lamda)
        calls = []
        
        for _ in range(n_calls):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)

            if poly.contains(Point(x, y)):
                calls.append((x, y))   # MISMO formato que ya usas

        call_stream.append(calls)
    return call_stream
    
# ============================================================
# AMBULANCES GENERATOR FUNCTION
# ============================================================
def generate_ambulances(num_amb, city_limits, G):
    """
    This function initializes ambulances' positions randomly within the city boundaries
    """
    poly = city_limits.union_all()
    xmin, ymin, xmax, ymax = poly.bounds
    ambulances = []
    # Fix random seed
    np.random.seed(42)
    while len(ambulances) < num_amb:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        p = Point(x, y)
        
        if poly.contains(p):
            i = len(ambulances)
            pos = (x, y) 
            start_node = ox.distance.nearest_nodes(G, x, y)
            ambulances.append(Ambulance(i, pos, start_node))
            
    return ambulances

def nearest_hospital(call, hospitals, G):
    """
    This function identifies the nearest hospital for a given node.
    """
    times = [network_time(call, h, G) for h in hospitals]
    idx = np.argmin(times)
    return hospitals[idx], times[idx]


# ============================================================
# AMBULANCES GENERATOR FUNCTION
# ============================================================
def build_km_matrices(greedy_results, opt_results, max_filter=100, n_grid=200):
    """
    Builds KM-style evaluation matrices for RT and WT.

    Returns:
        grid,
        greedy_rt_curves, opt_rt_curves,
        greedy_wt_curves, opt_wt_curves
    """

    # --------------------------
    # COMMON TIME GRID
    # --------------------------
    greedy_times = np.concatenate([
        greedy_results["RT"].values,
        greedy_results["WT"].values
    ])

    greedy_filtered = greedy_times[greedy_times <= max_filter]

    grid = np.linspace(
        greedy_filtered.min(),
        greedy_filtered.max(),
        n_grid
    )

    # --------------------------
    # RT CURVES
    # --------------------------
    greedy_rt_curves = [
        get_km_curve(g["RT"].values, grid)
        for _, g in greedy_results.groupby("sim_id")
    ]

    opt_rt_curves = [
        get_km_curve(g["RT"].values, grid)
        for _, g in opt_results.groupby("sim_id")
    ]

    # --------------------------
    # WT CURVES
    # --------------------------
    greedy_wt_curves = [
        get_km_curve(g["WT"].values, grid)
        for _, g in greedy_results.groupby("sim_id")
    ]

    opt_wt_curves = [
        get_km_curve(g["WT"].values, grid)
        for _, g in opt_results.groupby("sim_id")
    ]

    return (
        grid,
        np.array(greedy_rt_curves),
        np.array(opt_rt_curves),
        np.array(greedy_wt_curves),
        np.array(opt_wt_curves)
    )
# ============================================================
# FUNCTION TO PRINT METRICS
# ============================================================
def print_metrics(name, data):
    """
    This function prints relevant metrics
    """
    print(f"\n{name}")
    print("-" * len(name))

    print(f"Samples : {len(data)}")
    print(f"Mean    : {np.mean(data):.2f} min")
    print(f"Median  : {np.median(data):.2f} min")
    print(f"Std Dev : {np.std(data):.2f} min")
    print(f"Min     : {np.min(data):.2f} min")
    print(f"Max     : {np.max(data):.2f} min")


# ============================================================
# KAPLAN MEIER CURVES
# ============================================================
def get_km_curve(times, grid):
    kmf = KaplanMeierFitter()
    kmf.fit(times)
    sf = kmf.survival_function_

    # Interpolate common grid
    return np.interp(grid, sf.index, sf["KM_estimate"])


# ============================================================
# SELECT HOSPITALS RANDOMLY
# ============================================================
def select_hospitals_randomly(hospitals, num_hosp):
    """
    This function select N hospitals randomly from the set of existing hospitals in quebec given value providad "num_hosp"
    """
    hospitals_sample = hospitals.sample(n=num_hosp, random_state=42)
    hospitals_tuples = list(zip(hospitals_sample['lon'], hospitals_sample['lat']))
    hospital_nodes = [ox.distance.nearest_nodes(G, lon, lat) for lon, lat in hospitals_tuples]
    hospitals_precomputed = list(zip(hospitals_tuples, hospital_nodes))
    return hospitals_precomputed
    
# ============================================================
# SURVIVAL CURVES' COMPARISON
# ============================================================
def plot_band_compare(ax, curves_a, curves_b, title, grid, label_a="Greedy", label_b="Optimized", vlines=None):

    def stats(curves):
        median = np.median(curves, axis=0)
        p5 = np.percentile(curves, 5, axis=0)
        p95 = np.percentile(curves, 95, axis=0)
        return median, p5, p95

    def get_S_at_t(median, t):
        return np.interp(t, grid, median)
        
    # ---------------- A ----------------
    med_a, p5_a, p95_a = stats(curves_a)

    ax.plot(grid, med_a, color="red", linewidth=2)
    ax.fill_between(grid, p5_a, p95_a, color="red", alpha=0.15)

    # ---------------- B ----------------
    med_b, p5_b, p95_b = stats(curves_b)

    ax.plot(grid, med_b, color="black", linewidth=2)
    ax.fill_between(grid, p5_b, p95_b, color="gray", alpha=0.2)
    
    # ---------------- VERTICAL MARKERS ----------------
    if vlines is not None:
        for t in vlines:
            Sa = np.interp(t, grid, med_a)
            Sb = np.interp(t, grid, med_b)

            # vertical line
            ax.vlines(t, 0, max(Sa, Sb), linestyle="--", color="blue", alpha=0.5)
            ax.hlines(Sa, 0, t, colors="red", linestyles="--", alpha=0.5)
            ax.hlines(Sb, 0, t, colors="black", linestyles="--", alpha=0.5)
            ax.scatter([t, t], [Sa, Sb], color=["red", "black"], zorder=5)
  
    # PLOT        
    ax.set_title(title)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1)

    fig = ax.figure
    
    handles = [
        mlines.Line2D([], [], color="red", linewidth=2, label=f"{label_a} median"),
        mpatches.Patch(color="red", alpha=0.15, label=f"{label_a} 5–95%"),
        mlines.Line2D([], [], color="black", linewidth=2, label=f"{label_b} median"),
        mpatches.Patch(color="gray", alpha=0.2, label=f"{label_b} 5–95%")
    ]
    
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False
    )
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    # ---------------- metrics print ----------------
    if vlines is not None:
        print('--------------------------------------')
        print(f"Exceedance Probability ({title})")
        print('--------------------------------------')
        for t in vlines:
            Sa = get_S_at_t(med_a, t)
            Sb = get_S_at_t(med_b, t)

            print(f"P(t > {t} minutes)")
            print(f"  {label_a}: {100*Sa:.1f}%")
            print(f"  {label_b}: {100*Sb:.1f}%")
            
# ============================================================
# GREEDY POLICY
# ============================================================
def greedy_policy(calls, ambulances, hospitals_precomputed, t, queue):
    """
    This function carries out the greedy policy given the input data: number of emergency calls, open hospitals and ambulances.
    """
    response_times = []
    hold_on_times = []

    # 1. Add new calls at the end of the queue each timestep
    queue.extend(calls)

    # 2. Process the queue while there are hold-on calls and available ambulances
    while len(queue) > 0:
        
        # Verify available ambulances
        available = [a for a in ambulances if a.available]
        
        # if no ambulances -> calls will get in queue and will be processed at timestep t+1.
        if len(available) == 0:
            break

        # we process first calls first
        call = queue.pop(0)
        call_coords = call["coords"]
        t_creation = call["t_creation"]

        # WAITING TIME
        hold_on_time = t - t_creation
        hold_on_times.append(hold_on_time)
        
        # 3. Find the nearest ambulance to this call
        call_node = ox.distance.nearest_nodes(G, call_coords[0], call_coords[1])
        travel_times = [network_time(a.current_node, call_node, G) for a in available]
        
        idx = np.argmin(travel_times)
        amb = available[idx]

        # Time an ambulance takes to go to pick up the patient
        travel_time = travel_times[idx]
        
        # Response time = hold-on time + travel time the ambulance takes to reach the patient
        response_time = hold_on_time + travel_time
        response_times.append(response_time)
        
        # 4. Find nearest hospital
        h_times = [network_time(call_node, h_info[1], G) for h_info in hospitals_precomputed]
        h_idx = np.argmin(h_times)
        best_hospital_coords = hospitals_precomputed[h_idx][0]
        best_hospital_node = hospitals_precomputed[h_idx][1]
        hospital_time = h_times[h_idx]

        # 5. Traffic uncertainty 
        traffic_factor = np.max([1.0, np.random.normal(loc=1.2, scale=0.1)])
        total_travel = (travel_time + hospital_time) * traffic_factor

        # Assign mission to the ambulance
        amb.assign(best_hospital_coords, best_hospital_node, total_travel, t)
    
    return {
        'response_times': response_times,
        'hold_on_times': hold_on_times
    }



# ============================================================
# OPTIMIZATION POLICY
# ============================================================
def optimization_policy(calls, ambulances, hospitals_precomputed, t, queue):
    """
    This function carries out the optimization policy for a given number of emergency calls, hospitals and ambulances.
    """
    response_times = []
    hold_on_times = []
    
    queue.extend(calls)
    available = [a for a in ambulances if a.available]

    if len(available) == 0 or len(queue) == 0:
        return {
            'response_times': response_times,
            'hold_on_times': hold_on_times
        }

    current_queue_calls = list(queue)
    
    cost_matrix = compute_cost_matrix(available, current_queue_calls, G)
    #cost_matrix_safe = np.where(cost_matrix >= 9999.0, 999999.0, cost_matrix)

    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        return { 'response_times': response_times, 'hold_on_times': hold_on_times }

    assigned_calls_indices = sorted(list(col_ind), reverse=True)
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= 9999.0:
            if c in assigned_calls_indices:
                assigned_calls_indices.remove(c)
            continue

        amb = available[r]
        call_item = current_queue_calls[c]
        call_node = call_item["node"] 
        t_creation = call_item["t_creation"]

        hold_on_time = t - t_creation
        hold_on_times.append(hold_on_time)
        
        travel_time = cost_matrix[r, c]
        response_time = hold_on_time + travel_time
        response_times.append(response_time)
        
        h_times = [network_time(call_node, h_info[1], G) for h_info in hospitals_precomputed]
        h_idx = np.argmin(h_times)
        best_hospital_coords = hospitals_precomputed[h_idx][0]
        best_hospital_node = hospitals_precomputed[h_idx][1]
        hospital_time = h_times[h_idx]

        uncertanty = np.random.normal(loc=1.2, scale=0.1)
        traffic_factor = np.max([1.0, uncertanty])
        total_travel = (travel_time + hospital_time) * traffic_factor 

        amb.assign(best_hospital_coords, best_hospital_node, total_travel, t)

    for idx in assigned_calls_indices:
        queue.pop(idx)
        
    return {
        'response_times': response_times,
        'hold_on_times': hold_on_times
    }



# ============================================================
# SIMULATION ROUTINE
# ============================================================
def run_simulation(T, initial_ambulances, hospitals_precomputed, policy, sim_id=0, fixed_calls=None):
    """
    Run greedy or optimization policy and return full event-level DataFrame. 
    Each row = one attended call (patient-level event).
    """

    limits_raw = gpd.read_file(r"C:\Users\jesarauj001\OneDrive - INRS\Desktop\Geosatial Opimization Project\Ambulances\BOUNDARIES.shp") 
    limits_raw['district'] = limits_raw['ARS_NM_ARR']
    city_limits = limits_raw[['district','geometry']]

    # Hacemos una copia profunda de la flota inicial
    ambulances = copy.deepcopy(initial_ambulances)

    all_rows = []
    call_queue = []
    # TIME LOOP
    for t in range(T):

        # Update ambulance state
        for amb in ambulances:
            amb.update(t)

        # Generate calls
        calls = fixed_calls[t] #generate_calls(lamda, city_limits)
        
        calls_with_time = [] 
        for single_call in calls:
            node = ox.distance.nearest_nodes(G, single_call[0], single_call[1])
            calls_with_time.append({
                'coords':single_call, 
                't_creation':int(t), 
                'node':int(node)
            })

        # Apply policy
        if policy == "greedy":
            rt = greedy_policy(calls_with_time, ambulances, hospitals_precomputed, t, call_queue)
        elif policy == "optimization":
            rt = optimization_policy(calls_with_time, ambulances, hospitals_precomputed, t, call_queue)
        else:
            raise ValueError("Invalid policy")

        # Extract values
        n_events = len(rt['response_times'])

        for i in range(n_events):
            all_rows.append({
                "sim_id": sim_id,
                "policy": policy,
                "time": t,
                "RT": rt['response_times'][i],
                "WT": rt['hold_on_times'][i],
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    return df

# ----------------------------------
# MONTECARLO SIMULATION
# ---------------------------------
# 1. El puente que conecta los hilos con la simulación respetando tus argumentos
def _worker_simulation(sim_id, T, initial_ambulances, lamda, hospitals_precomputed, policy):
    return run_simulation(T, initial_ambulances, lamda, hospitals_precomputed, policy, sim_id=sim_id)
    
def _worker_simulation_crn(sim_id, T, initial_ambulances, hospitals_precomputed, lamda, city_limits):
    fixed_calls = generate_call_stream(T, lamda, city_limits, seed=sim_id)
    greedy = run_simulation(T, initial_ambulances, hospitals_precomputed, "greedy", sim_id, fixed_calls)
    opt = run_simulation(T, initial_ambulances, hospitals_precomputed, "optimization", sim_id, fixed_calls)
    return greedy, opt
    
# 2. El motor de Monte Carlo en paralelo con barra de progreso y tiempo restante 
def monte_carlo_parallel(N_SIM, initial_ambulances, lamda, hospitals_precomputed, T, city_limits, max_workers=None):

    greedy_results = []
    opt_results = []

    print('-----------------------------------------')
    print('Starting Parallel Monte Carlo (CRN)')
    print('-----------------------------------------')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(_worker_simulation_crn,
                            sim,
                            T,
                            initial_ambulances,
                            hospitals_precomputed,
                            lamda,
                            city_limits): sim
            for sim in range(N_SIM)
        }

        for future in tqdm(as_completed(futures), total=N_SIM, desc="Simulando CRN"):

            try:
                greedy, opt = future.result()
                greedy_results.append(greedy)
                opt_results.append(opt)

            except Exception as e:
                print(f"[!] Error: {e}")

    return pd.concat(greedy_results, ignore_index=True), pd.concat(opt_results, ignore_index=True)
    
# ----------------------------------------
# DELTA SURVIVAL PLOT
# ----------------------------------------
def plot_delta_survival(
    opt_rt_curves,
    greedy_rt_curves,
    opt_wt_curves,
    greedy_wt_curves,
    grid
):

    def delta_stats(opt_curves, greedy_curves):

        n = min(len(opt_curves), len(greedy_curves))

        delta = opt_curves[:n] - greedy_curves[:n]

        median = np.median(delta, axis=0)
        p5 = np.percentile(delta, 5, axis=0)
        p95 = np.percentile(delta, 95, axis=0)
        std = np.std(delta, axis=0)

        return median, p5, p95, std

    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    # ---------------- RT ----------------
    med, p5, p95, std = delta_stats(opt_rt_curves,
                               greedy_rt_curves)

    axes[0].plot(grid, med, color='black', lw=2,
                 label='Median ΔS(t)')

    axes[0].fill_between(grid, p5, p95,
                         color='blue',
                         alpha=0.3,
                         label='5–95%')

    axes[0].fill_between(grid, 
                         med-std/2, 
                         med+std/2,
                         color='green',
                         alpha=0.3,
                         label='std')
    
    axes[0].axhline(0,
                    color='red',
                    ls='--',
                    lw=1)

    axes[0].set_title('Δ Response Time')
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Δ Survival Probability')

    # ---------------- WT ----------------
    med, p5, p95, std = delta_stats(opt_wt_curves,
                               greedy_wt_curves)

    axes[1].plot(grid, med, color='black', lw=2,
                 label='Median ΔS(t)')

    axes[1].fill_between(grid, p5, p95,
                         color='blue',
                         alpha=0.3,
                         label='5–95%')

    axes[1].fill_between(grid, 
                         med-std/2, 
                         med+std/2,
                         color='green',
                         alpha=0.3,
                         label='std')

    axes[1].axhline(0,
                    color='red',
                    ls='--',
                    lw=1)

    axes[1].set_title('Δ Waiting Time')
    axes[1].set_xlabel('Time (min)')

    # leyenda global
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles,
               labels,
               loc='lower center',
               ncol=3,
               frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    plt.show()


def empirical_survival(x, grid):
    """
    S(t)=P(X>t)
    """
    x = np.asarray(x)
    return np.array([(x > t).mean() for t in grid])
    
# =========================
# PERFORMANCE PLOTS
# =========================
def performance_plots(greedy_results,opt_results):
    """
    This function plot ΔS curves, AUC and maximum time gains and  tail probabilities for response and waiting times.
    """
    
    # Remove outliers
    greedy_results = greedy_results[(greedy_results['RT']<100) & (greedy_results['WT']<100)]
    opt_results = opt_results[(opt_results['RT']<100) & (opt_results['WT']<100)]
    
    ambulance_levels = sorted(greedy_results["A"].unique())
    
    results_summary = []
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    # =========================
    # LOOP RT / WT SHARED LOGIC
    # =========================
    
    def compute_metrics_and_plot(metric, row, col):
        for A in ambulance_levels:
    
            g = greedy_results.loc[greedy_results["A"] == A, metric]
            o = opt_results.loc[opt_results["A"] == A, metric]
    
            t_max = max(g.max(), o.max())
            grid = np.linspace(0, t_max, 200)
    
            Sg = empirical_survival(g, grid)
            So = empirical_survival(o, grid)
    
            deltaS = So - Sg 
    
            auc = np.trapz(deltaS, grid)
    
            idx_min = np.argmin(deltaS)
            t_star = grid[idx_min]
            delta_min = deltaS[idx_min]
    
            results_summary.append({
                "A": A,
                "metric": metric,
                "ΔS_min": delta_min,
                "ΔS_tmin": t_star,
                "AUC_ΔS": auc
            })
    
            axes[row, col].plot(grid, deltaS*100, label=f"A={A}")
    
        axes[row, col].axhline(0, color="black", ls="--")
        axes[row, col].set_title(f"ΔS(t) - {metric}")
        axes[row, col].set_xlabel("Minutes")
        axes[row, col].set_ylabel("%")
        axes[row, col].grid(alpha=0.3)
        axes[row, col].legend(title="N_amb")
    
    
    # =========================
    # ROW 1: RT
    # =========================
    compute_metrics_and_plot("RT", 0, 0)
    
    # AUC RT
    A_vals = sorted(set(r["A"] for r in results_summary if r["metric"] == "RT"))
    AUC_RT = [r["AUC_ΔS"] for r in results_summary if r["metric"] == "RT"]
    
    axes[0,1].bar(A_vals, AUC_RT)
    axes[0,1].set_title("Global Benefit(AUC)")
    axes[0,1].set_ylabel("Minutes")
    axes[0,1].set_xlabel("Number of ambulances")
    axes[0,1].grid(alpha=0.3)
    
    # Peak RT
    peak_RT = [r["ΔS_min"] * -100 for r in results_summary if r["metric"] == "RT"]
    
    axes[0,2].bar(A_vals, peak_RT)
    axes[0,2].set_title("Maximum Benefit - $ΔS_{min}(t)$")
    axes[0,2].set_xlabel("Number of ambulances")
    axes[0,2].set_ylabel("%")
    axes[0,2].grid(alpha=0.3)

    rt_greedy_tail = greedy_results.groupby("A")["RT"].apply(lambda x: (x > 25).mean())
    rt_opt_tail    = opt_results.groupby("A")["RT"].apply(lambda x: (x > 25).mean())

    # Tail RT
    axes[0,3].plot(rt_greedy_tail.index, rt_greedy_tail.values*100, marker="o", label="GP")
    axes[0,3].plot(rt_opt_tail.index, rt_opt_tail.values*100, marker="o", label="OP")
    axes[0,3].set_title("P(RT > 25)")
    axes[0,3].set_xlabel("Number of ambulances")
    axes[0,3].set_ylabel("%")
    axes[0,3].legend()
    axes[0,3].grid(alpha=0.3)
    
    # =========================
    # ROW 2: WT
    # =========================
    compute_metrics_and_plot("WT", 1, 0)
    
    # AUC WT
    AUC_WT = [r["AUC_ΔS"] for r in results_summary if r["metric"] == "WT"]
    axes[1,1].bar(A_vals, AUC_WT)
    axes[1,1].set_title("Global Benefit (AUC)")
    axes[1,1].set_xlabel("Number of ambulances")
    axes[1,1].set_ylabel("Minutes")
    axes[1,1].grid(alpha=0.3)
    
    # Peak WT
    peak_WT = [r["ΔS_min"] * -100 for r in results_summary if r["metric"] == "WT"]
    axes[1,2].bar(A_vals, peak_WT)
    axes[1,2].set_title("Maximum Benefit - $ΔS_{min}(t)$")
    axes[1,2].set_xlabel("Number of ambulances")
    axes[1,2].set_ylabel("%")
    axes[1,2].grid(alpha=0.3)
    
    # Tail WT
    wt_greedy_tail = greedy_results.groupby("A")["WT"].apply(lambda x: (x > 10).mean())
    wt_opt_tail    = opt_results.groupby("A")["WT"].apply(lambda x: (x > 10).mean())
    
    axes[1,3].plot(wt_greedy_tail.index, wt_greedy_tail.values*100, marker="o", label="GP")
    axes[1,3].plot(wt_opt_tail.index, wt_opt_tail.values*100, marker="o", label="OP")
    axes[1,3].set_ylabel("%")
    axes[1,3].set_xlabel("Number of ambulances")
    axes[1,3].set_title("P(WT > 10)")
    axes[1,3].legend()
    axes[1,3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()