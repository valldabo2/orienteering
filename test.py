import pandas as pd
import json
from itertools import product
from ortools.linear_solver import pywraplp
from dataclasses import dataclass
from typing import List

"""
https://support.gurobi.com/hc/en-us/community/posts/4404976974865-Orienteering-Problem
http://www.mysmu.edu/faculty/hclau/doc/EJOR%20-%20Orienteering%20Survey.pdf

"""

@dataclass
class Solution:
    path: List
    value: float
    cost: float
    solver: pywraplp.Solver


def get_cost(costs, i, j, nodes_mapping):
    c = (
        costs
        .loc[
            lambda df:
            (df.src == nodes_mapping[i]) & (df.dest == nodes_mapping[j])
        ]
        .cost
    )
    if len(c) > 0:
        return float(c.iloc[0])
    else:
        return 0


def solve_orienteering(costs, rewards, start_end_node, max_cost):
    nodes_mapping = {i: n for i, n in enumerate(rewards.node, 1)}
    start_node_index = list(nodes_mapping.keys())[list(nodes_mapping.values()).index(start_end_node)]

    # Rewards dict
    s_i = {
        i:
        float(
            rewards
            .loc[lambda df: df.node == nodes_mapping[i]]
            .reward
            .iloc[0]
        )
        for i in nodes_mapping
    }

    # Create costs the other way round
    reversed = (
        costs
        .copy()
        .rename(columns={"src": "dest", "dest": "src"})
    )
    costs = pd.concat([costs, reversed], axis=0)

    # Create inidices and costs
    indexes = list(nodes_mapping.keys())
    t_ij = {
        i:
        {
            j: get_cost(costs, i, j, nodes_mapping)
            for j in indexes
        }
        for i in indexes
    }

    index_2_N = indexes[1:]
    N = indexes[-1]

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

   # Matrix that will store 1 if there should be connection between nodes, 0 otherwise
   # That's one of the things we'll be optimizing
    x_ij = {
        i:{
            j: solver.IntVar(0.0, 1.0, f"x_{i}_{j}")
            for j in indexes
        } 
        for i in indexes 
    }

    # Matrix that will store the order of node visits
    # Other things that gets optimized
    u_i = {
        i: solver.IntVar(0.0, solver.infinity(), f"u_{i}")
        for i in nodes_mapping
    }

    # Node 1 outgoing - must
    solver.Add(sum(x_ij[start_node_index][j] for j in index_2_N) == 1)
    # Node 1 incoming - must
    solver.Add(sum(x_ij[i][start_node_index] for i in index_2_N) == 1)

    # Other incoming and outcoming
    for k in index_2_N:

        # There should be no more than one outgoing connection from each node
        solver.Add(
            sum(x_ij[i][k] for i in indexes) <= 1
        )
        # There should be no more than one incomming connection from each node
        solver.Add(
           sum(x_ij[k][j] for j in indexes) <= 1 
        )
        # If there is incomming there must be outgoing
        solver.Add(
            sum(x_ij[i][k] for i in indexes) == sum(x_ij[k][j] for j in indexes)
        )

    # Prevent subtours (Miller et al. 1960).
    for i in index_2_N:
        solver.Add(u_i[i] <= N)
        solver.Add(u_i[i] >= 2)
        for j in indexes:
            solver.Add(u_i[i] - u_i[j] + 1 <= (N -1)*(1-x_ij[i][j]))

    # Max cost
    solver.Add(
        sum(t_ij[i][j] * x_ij[i][j] for i,j in product(indexes, indexes)) <= max_cost
    )   

    # Maximization
    solver.Maximize(sum(s_i[i] * x_ij[i][j] for i,j in product(indexes, indexes)))

    def cost(t_ij, x_ij):
        return sum(t_ij[i][j] * x_ij[i][j] for i,j in product(indexes, indexes))
    
    def real_cost(t_ij, x_ij):
        x_ij = {i: {j: x_ij[i][j].solution_value() for j in x_ij[i]} for i in x_ij}
        return cost(t_ij, x_ij)

    solver.Solve()

    # Format output
    variables = {(i,j): x_ij[i][j] for i in x_ij for j in x_ij[i]}
    used = {ij: xij for ij, xij in variables.items() if xij.solution_value() == 1.0}
    used_edges = [(nodes_mapping[i], nodes_mapping[j]) for (i,j) in used]

    path = extract_path(used_edges)

    return Solution(
        list(path),
        solver.Objective().Value(),
        real_cost(t_ij, x_ij),
        solver
    )


def extract_path(edges):
    savedEdge = (None, None)
    for i, e in enumerate(edges):
        if i == 0:
            yield e[0]
            yield e[1]
            savedEdge = e
        else:
            for j, ed in enumerate(edges):
                if edges[j][0] == savedEdge[1]:
                    savedEdge = ed
                    yield edges[j][1]
        

def print_solution(x_ij, solver, nodes_mapping):
    for i in x_ij:
        for j in x_ij[i]:
            v =  x_ij[i][j].solution_value()
            if v == 1.0:
                print(i,j, nodes_mapping[i], nodes_mapping[j], v)
    print(solver.Objective().Value())


def test_orienteering_problem_1():
    edges = pd.DataFrame([
        ["a", "b", 2],
        ["a", "c", 50],
        ["a", "d", 1],
        ["b", "c", 20],
        ["b", "d", 2],
        ["c", "d", 20],
    ], columns=["src", "dest", "cost"])

    nodes = pd.DataFrame([
        ["a", 0],
        ["b", 11],
        ["c", 5],
        ["d", 10]
    ], columns=["node", "reward"])

    solution = solve_orienteering(edges, nodes, "a", 5)
    #assert solution.path == ["a", "b", "d", "a"]
    assert solution.value == 21
    assert solution.cost == 5

    solution = solve_orienteering(edges, nodes, "a", 4)
    assert solution.path == ["a", "b", "a"]
    assert solution.value == 11
    assert solution.cost == 4

    solution = solve_orienteering(edges, nodes, "a", 2)
    assert solution.path == ["a", "d", "a"]
    assert solution.value == 10
    assert solution.cost == 2

    solution = solve_orienteering(edges, nodes, "a", 43)
    #assert solution.path == ["a", "b", "c", "d", "a"]
    assert solution.value == 26
    assert solution.cost == 43


test_orienteering_problem_1()
