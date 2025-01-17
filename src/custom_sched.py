import copy
import logging
import os
import random
import time

# from graph import find_longest_path_dfs
from processor import Core
from rta_alphabeta_new import (
    Eligiblity_Ordering_PA,
    EO_iter,
    find_providers_consumers,
    load_task,
)
from task import DAGTask, Job

EXECUTION_MODEL = [
    "WCET",
    "HALF_RANDOM",
    "HALF_RANDOM_NORM",
    "FULL_RANDOM",
    "FULL_RANDOM_NORM",
    "BCET",
]
PREEMPTION_COST = 0
MIGRATION_COST = 0

PATH_OF_SRC = os.path.dirname(os.path.abspath(__file__))
LOG_TO_FILE_LOCATION = PATH_OF_SRC + "/../results/log.txt"


def find_all_paths(G_, start_vertex, end_vertex, path=[]):
    """find all paths from start_vertex to end_vertex in graph"""
    graph = G_
    path = path + [start_vertex]

    if start_vertex == end_vertex:
        return [path]

    if start_vertex not in graph:
        return []

    paths = []
    for vertex in graph[start_vertex]:
        if vertex not in path:
            # solve this in a recursive way
            extended_paths = find_all_paths(G_, vertex, end_vertex, path)
            for p in extended_paths:
                paths.append(p)

    return paths


def find_longest_path_dfs(G_, start_vertex, end_vertex, weights):
    """find the longest path with depth first search"""

    # find all paths
    paths = find_all_paths(G_, start_vertex, end_vertex)

    # search for the critical path
    costs = []
    for path in paths:
        cost = 0
        for v in path:
            cost = cost + weights[v - 1]
        costs.append(cost)

    (m, i) = max((v, i) for i, v in enumerate(costs))

    return (m, paths[i])


def trace_init(log_to_file=False, debug=False):
    LOG_FORMAT = "[%(asctime)s-%(levelname)s: %(message)s]"
    LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

    log_mode = logging.DEBUG if debug else logging.INFO

    if log_to_file:
        logging.basicConfig(
            filename="log.txt",
            filemode="a",
            level=log_mode,
            format=LOG_FORMAT,
            datefmt=LOG_DATEFMT,
        )
    else:
        logging.basicConfig(level=log_mode, format=LOG_FORMAT, datefmt=LOG_DATEFMT)


def trace(msglevel, timestamp, message):
    msg = f"t = {timestamp}: {message}"
    if msglevel == 0:
        logging.debug(msg)
    elif msglevel == 1:
        logging.info(msg)
    elif msglevel == 2:
        logging.warning(msg)
    elif msglevel == 3:
        logging.error(msg)


def Eligiblity_Ordering_PA(G_dict, C_dict):

    Prio = {}

    # --------------------------------------------------------------------------
    # I. load task parameters
    C_exp = []
    for key in sorted(C_dict):
        C_exp.append(C_dict[key])

    V_array = list(copy.deepcopy(G_dict).keys())
    V_array.sort()
    _, lamda = find_longest_path_dfs(G_dict, V_array[0], V_array[-1], C_exp)

    VN_array = V_array.copy()

    for i in lamda:
        if i in VN_array:
            VN_array.remove(i)

    # --------------------------------------------------------------------------
    # II. initialize eligbilities to -1
    for i in G_dict:
        Prio[i] = -1

    # --------------------------------------------------------------------------
    # III. providers and consumers
    # iterative all critical nodes
    # after this, all provides and consumers will be collected

    # >> for time measurement
    global time_EO_CPC
    begin_time = time.time()
    # << for time measurement

    providers, consumers = find_providers_consumers(G_dict, lamda, VN_array)

    # >> for time measurement
    time_EO_CPC = time.time() - begin_time
    # << for time measurement

    # --------------------------------------------------------------------------
    # IV. Start iteration
    # >> for time measurement
    global time_EO
    begin_time = time.time()
    # << for time measurement

    EO_iter(G_dict, C_dict, providers, consumers, Prio)

    # >> for time measurement
    time_EO = time.time() - begin_time
    # << for time measurement

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for i in Prio:
        if Prio[i] <= 1:
            pass
            # raise Exception("Some prioirities are not assigned!")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return Prio


def sched(dag, algorithm="random", execution_model="WCET"):
    """
    Policies:
    - random (dynamic)
    - eligibility
    - TPDS2019
    - EMSOFT2019 (dynamic)

    Execution models:
    - WCET
    - half_random
    - full_random
    """
    T_MAX = 1_000_000
    t = 0

    core = Core()
    w_queue = dag.V.copy()  # Waiting queue
    r_queue = []  # Ready queue
    f_queue = []  # Finished queue

    Prio = Eligiblity_Ordering_PA(dag.G, dag.C_dict)

    trace(0, t, f"Algorithm = {algorithm}, Exe_Model = {execution_model}, #Cores = 1")

    r_queue.append(1)
    w_queue.remove(1)

    while t < T_MAX and f_queue != dag.V:
        trace(0, t, "Scheduling point reached!")

        # Update ready queue
        for node in w_queue.copy():
            if all(pred in f_queue for pred in dag.pre[node]):
                r_queue.append(node)
                w_queue.remove(node)

        # Assign task to core if idle
        if core.idle and r_queue:
            task_idx = max(r_queue, key=lambda i: Prio[i])
            task_wcet = dag.C[task_idx - 1]
            tau = Job(idx_=task_idx, C_=task_wcet)
            core.assign(tau)
            r_queue.remove(task_idx)
            trace(0, t, f"Job {task_idx} assigned to Core 1")

        # Determine next scheduling point
        sp = min(core.get_workload() or float("inf"), 1)

        # Execute tasks
        t += sp
        tau_idx, tau_finished = core.execute(sp)

        if tau_finished:
            f_queue.append(tau_idx)
            trace(0, t, f"Job {tau_idx} finished on Core 1")

    makespan = t

    if t < T_MAX:
        trace(0, t, f"Finished: Makespan is {makespan}")
    else:
        trace(3, t, "Simulation Overrun!")

    return makespan


if __name__ == "__main__":
    trace_init(log_to_file=True)

    G_dict, C_dict, C_array, lamda, VN_array, L, W = load_task(0)
    dag = DAGTask(G_dict, C_array)

    R1 = sched(dag, algorithm="eligibility", execution_model="WCET")

    print("Experiment finished!")
