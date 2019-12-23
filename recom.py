from math import floor, ceil
from collections import deque
from gerrychain import Graph, Partition
from networkx.readwrite import json_graph
from julia.api import Julia
try:
    _jl = Julia(sysimage='sys.so', compiled_modules=False)
except RuntimeError:
    _jl = Julia(compiled_modules=False)
from julia import Main, Random, Flips  # pylint: disable=E402


class RecomStep:
    """A step in a ReCom chain (with multiplicity)."""
    def __init__(self, partition: Partition, multiplicity: int, meta=None):
        self.partition = partition
        self.multiplicity = multiplicity
        self.meta = meta


class RecomChain:
    """A fast Julia-based reversible ReCom chain implementation."""
    def __init__(self, graph: Graph, initial_state: Partition,
                 total_steps: int, pop_col: str, pop_tol: float,
                 compressed: bool = True, reversible: bool = True,
                 seed: int = None):
        """
        :param graph: the GerryChain graph to compile to a Julia graph.
        :param initial_state: The state to start from.
        :param total_steps: The total steps to take in the chain.
        :param pop_col: The column of the graph with population data.
        :param pop_tol: The population tolerance to use.
        :param compressed: If `True`, plans are represented in `RecomStep`
            format. Otherwise, behavior mimics GerryChain.
        :param reversible: Determines if the reversible version of the chain
            should be used.
        :param seed: The random seed to use for plan generation.
        """
        for node, assignment in initial_state.assignment.items():
            graph.nodes[node]['__district'] = assignment
        graph_data = json_graph.adjacency_data(graph)
        self.graph = Flips.IndexedGraph(graph_data, pop_col)
        # TODO: verify initial state is valid w.r.t. population constraints
        #       (+ more?)
        self.plan = Flips.Plan(self.graph, '__district')
        total_pop = sum(graph.nodes[node][pop_col] for node in graph.nodes)
        districts = list(initial_state.assignment.values())
        n_districts = max(districts) - min(districts) + 1
        district_pop = total_pop / n_districts
        self.min_pop = int(ceil((1 - pop_tol) * district_pop))
        self.max_pop = int(floor((1 + pop_tol) * district_pop))
        self.reversible = reversible
        self.compressed = compressed

        self._curr_parent = initial_state
        self.total_steps = total_steps
        self._curr_step = 0
        self._buf = deque()  # https://stackoverflow.com/a/4426727/8610749
        self.data = None
        if seed is None:
            self._twister = Random.MersenneTwister()
        else:
            self._twister = Random.MersenneTwister(seed)

    def __iter__(self):
        return self

    def __next__(self) -> Partition:
        """Returns the next Partition in the chain."""
        if self._curr_step > self.total_steps:
            raise StopIteration
        if not self._buf:
            chain_run = Flips.pychain(self.graph,
                                      self.plan,
                                      self.total_steps,
                                      self.reversible,
                                      self.min_pop,
                                      self.max_pop,
                                      self._twister)
            self.data = chain_run
            for step in chain_run:
                partition_cls = self._curr_parent.__class__
                self._curr_parent = partition_cls(parent=self._curr_parent,
                                                  flips=step['flip'])
                if self.compressed:
                    step_meta = {
                        'reasons': step['reasons'],
                        'district_adj': step['district_adj']
                    }
                    self._buf.append(RecomStep(self._curr_parent,
                                               step['self_loops'] + 1,
                                               step_meta))
                else:
                    for _ in range(step['self_loops']):
                        # Add self-loops (identical plans) to buffer
                        self._buf.append(self._curr_parent)
                    self._buf.append(self._curr_parent)
        step = self._buf.popleft()
        self._curr_step += step.multiplicity
        return step
