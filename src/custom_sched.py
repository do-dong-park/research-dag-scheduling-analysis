import logging
import os
import random

from processor import Core
from rta_alphabeta_new import Eligiblity_Ordering_PA, load_task
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
