
from egglog.exp.param_eq.pipeline import *
from egglog.exp.param_eq.pipeline import _graph_size

TIMES: list[tuple[str, float]] = []

def mark_time(event: str):
    TIMES.append((event, time.perf_counter()))

def print_times():
    """
    Prints out % of time spent in each phase of the pipeline, as well as total time.
    """
    print("Pipeline timings:")
    total_time = TIMES[-1][1] - TIMES[0][1]
    for i in range(1, len(TIMES)):
        event, t = TIMES[i]
        _prev_event, prev_t = TIMES[i - 1]
        duration = t - prev_t
        print(f"{event}: {duration:.2f}s ({(duration / total_time) * 100:.2f}%)")
    print(f"Total time: {total_time:.2f}s")

container_schedule.__egg_decls__
containers_analysis_schedule.__egg_decls__

# EXPR = "exp(1.24138165845371 * (log(abs(x0)) - exp(exp(log(abs(x0)) * exp(-0.0146126519824622 * exp(9.0 * log(abs(x1))))) / log(abs(log(abs((x1 ** 6.0 + log(abs(log(abs((log(abs(x0)) + log(abs((log(abs((log(abs(-1.0 * ((x1 ** 9.0 + 0.441) ** 9.0 + 0.24) ** 3.0)) + 0.292) ** 3.0)) + 0.462550286672301) ** 3.0))) ** 3.0)) ** 3.0))) ** 3.0)) ** 3.0)))) ** 3.0 * exp(-1.0 * log(abs(x1)))) + 0.904"
EXPR = "0.02889 * x0 ** 3.0 - 0.18081288 * x0 ** 2.0 + 0.18081288 * x0 + 0.14418036 * x1 - 0.00963 * exp(x1) + 0.179028 - 0.00963 * exp(-15.767 * x0)"
PARSED = parse_expression(EXPR)
CONTAINER = binary_to_containers(PARSED)

current, before_cost = EGraph().extract(CONTAINER, include_cost=True, cost_model=container_cost_model)
mark_time("start")

start = time.perf_counter()
# Add constants to the egraph so that they can be used in rules without needing to be registered each pass
egraph = EGraph(Num(0.0), save_egglog_string=False)
mark_time("create e-graph")

egraph._add_decls(containers_analysis_schedule)
mark_time("initial add decls")
# egraph._state._total_time = 0
cmd = egraph._state.run_schedule_to_egg(containers_analysis_schedule.schedule)
# print(f"time spent actually running egg: {egraph._state._total_time:.2f}s")
# del egraph._state._total_time
mark_time("initial schedule to egg")
egraph._state.run_program(cmd)
mark_time("initial run")

last_cost = before_cost
max_size = 0
passes = 0
for pass_index in range(1, MAX_PASSES + 1):
    with egraph:
        # egraph._state._total_time = 0
        n = egraph.let("n", current)
        mark_time(f"pass {pass_index} let")
        previous_size = _graph_size(egraph)
        for _i in range(HASKELL_INNER_ITERATION_LIMIT):
            egraph.run(containers_analysis_schedule)
            egraph.run(container_schedule)
            current_size = _graph_size(egraph)
            if current_size == previous_size:
                break
            previous_size = current_size
        mark_time(f"pass {pass_index} run")
        extracted, cost = egraph.extract(n, include_cost=True, cost_model=container_cost_model)
        mark_time(f"pass {pass_index} extract")
        # print(f"time spent actually running egg: {egraph._state._total_time:.2f}s")
    mark_time(f"pass {pass_index}")

    max_size = max(max_size, previous_size)
    passes = pass_index
    if extracted == current:
        break
    current = extracted
PaperPipelineReport(
    passes=passes,
    total_sec=time.perf_counter() - start,
    total_size=max_size,
    before_nodes=before_cost.node_count,
    before_params=before_cost.floats,
    extracted_nodes=last_cost.node_count,
    extracted_params=last_cost.floats,
    extracted=render_num(containers_to_binary(current)),
)


mark_time("end")


print_times()
