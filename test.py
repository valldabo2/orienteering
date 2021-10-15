import pandas as pd
import json
from itertools import product
from ortools.linear_solver import pywraplp

"""
https://support.gurobi.com/hc/en-us/community/posts/4404976974865-Orienteering-Problem
http://www.mysmu.edu/faculty/hclau/doc/EJOR%20-%20Orienteering%20Survey.pdf

"""

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
        return c.iloc[0]
    else:
        return 0

    
def solve_orienteering(costs, rewards, start_and_end, max_cost):
    nodes_mapping = {i: n for i, n in enumerate(rewards.node, 1)}
    # Adds end node
    nodes_mapping[len(nodes_mapping) + 1] = start_and_end
    s_i = {
        i:
        rewards.loc[lambda df: df.node == nodes_mapping[i]].reward.iloc[0]
        for i in nodes_mapping
    }

    reversed = (
        costs
        .copy()
        .rename(columns={"src": "dest", "dest": "src"})
    )
    costs = pd.concat([costs, reversed], axis=0)
    
    indexes = list(nodes_mapping.keys())
    t_ij = {
        i:
        {
            j: get_cost(costs, i, j, nodes_mapping)
            for j in indexes
        }
        for i in indexes
    }

    # print(nodes_mapping)
    # print(s)
    # print(json.dumps(t, indent=4, default=str))

    index_2_N = indexes[1:]
    index_1_N_1 = indexes[:-1]
    N = indexes[-1]
    print(index_2_N, index_1_N_1, N)


    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
   
    x_ij = {
        i:{
            j: solver.IntVar(0.0, 1.0, f"x_{i}_{j}")
            for j in indexes
        } 
        for i in indexes 
    }

    # Node 1 outgoing
    solver.Add(sum(x_ij[1][j] for j in index_2_N) == 1)
    # Node N incoming
    solver.Add(sum(x_ij[i][N] for i in index_1_N_1) == 1)

    # Other incoming and outcoming
    for k in index_2_N:
        solver.Add(
            sum(x_ij[i][k] for i in index_1_N_1) == sum(x_ij[k][j] for j in index_2_N)
        )

        solver.Add(
            sum(x_ij[i][k] for i in index_1_N_1) <= 1
        )
        solver.Add(
           sum(x_ij[k][j] for j in index_2_N) <= 1 
        )

    # Max cost
    solver.Add(
        sum(t_ij[i][j] * x_ij[i][j] for i,j in product(index_1_N_1, index_2_N)) <= max_cost
    )


    # Maximization
    solver.Maximize(sum(s_i[i] * x_ij[i][j] for i,j in product(index_1_N_1, index_2_N)))

    status = solver.Solve()
    print(nodes_mapping)
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal")
        for i,j in product(indexes, indexes):
            print(nodes_mapping[i],nodes_mapping[j], x_ij[i][j].solution_value())
        print(solver.Objective().Value())
    return



def test_orienteering_problem():
    edges = pd.DataFrame([
        ["a", "b", 1],
        ["a", "c", 20],
        ["a", "d", 1],
        ["b", "c", 20],
        ["b", "d", 1],
        ["c", "d", 20],
    ], columns=["src", "dest", "cost"])

    nodes = pd.DataFrame([
        ["a", 0],
        ["b", 10],
        ["c", 5],
        ["d", 10]
    ], columns=["node", "reward"])

    solution = solve_orienteering(edges, nodes, "a", 5)

    # assert solution.path == ["a", "b", "d", "a"]
    # assert solution.cost == 3
    # assert solution.reward == 20


test_orienteering_problem()
