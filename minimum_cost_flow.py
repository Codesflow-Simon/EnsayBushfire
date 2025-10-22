#!/usr/bin/env python3
"""
Minimum Cost Flow Problem for Bushfire Evacuation
Based on the mathematical formulation with finite supply and unbounded demand.

This script implements a network flow optimization to route population from 
population centers (positive population) to safe zones (negative population)
while minimizing total cost and respecting road capacities.
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
from data_interface import DataInterface

# Constants
POPULATION_PER_CAR = 1.0 # Average number of people per vehicle
EXISTING_TRAFFIC = 0.0 # Existing traffic as a fraction of the road capacity

def getCost(node1: str, node2: str, data_interface: DataInterface) -> float:
    """
    Get cost between two nodes. Currently returns road risk.
    
    Args:
        node1: First node name
        node2: Second node name
        data_interface: DataInterface instance
        
    Returns:
        Cost value (road risk)
        
    Raises:
        ValueError: If nodes are not connected
    """
    risk = data_interface.get_property(node1, node2, 'road_risk')
    time = data_interface.get_property(node1, node2, 'road_time_minutes')
    distance = data_interface.get_property(node1, node2, 'road_length')
    return risk * (time + distance)


def solve(data_interface):
    """Solve minimum cost flow problem and return result dict."""
    # Get node data directly from data interface
    population_nodes = data_interface.get_population_nodes()
    safe_nodes = data_interface.get_safe_nodes()
    intermediary_nodes = data_interface.get_intermediary_nodes()
    
    # Node supplies (convert population to car flow)
    node_supplies = {}
    for node in population_nodes + safe_nodes:
        population = data_interface.get_population(node)
        node_supplies[node] = population / POPULATION_PER_CAR  # Convert to car flow
    
    # Get all edges
    edges = data_interface.get_all_edges()
    
    # Create edge index mapping
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    n_vars = len(edges)
    
    # Objective: minimize total cost (road risk)
    c = np.array([getCost(edge[0], edge[1], data_interface) for edge in edges])
    
    # Constraints
    A_eq = []
    b_eq = []
    
    # Flow conservation for population centers (car flow)
    for node in population_nodes:
        constraint = np.zeros(n_vars)
        supply = node_supplies.get(node, 0)
        
        for edge in edges:
            idx = edge_to_idx[edge]
            if edge[0] == node:  # Outgoing
                constraint[idx] = 1
            elif edge[1] == node:  # Incoming
                constraint[idx] = -1
                
        A_eq.append(constraint)
        b_eq.append(supply)
        
    # Flow conservation for intermediary nodes
    for node in intermediary_nodes:
        constraint = np.zeros(n_vars)
        
        for edge in edges:
            idx = edge_to_idx[edge]
            if edge[0] == node:  # Outgoing
                constraint[idx] = 1
            elif edge[1] == node:  # Incoming
                constraint[idx] = -1
                
        A_eq.append(constraint)
        b_eq.append(0)
        
    # Safe absorption - total flow into safe zones equals total supply
    safe_constraint = np.zeros(n_vars)
    for node in safe_nodes:
        for edge in edges:
            idx = edge_to_idx[edge]
            if edge[1] == node:  # Incoming to safe zone
                safe_constraint[idx] = 1
                
    A_eq.append(safe_constraint)
    b_eq.append(sum(node_supplies.get(node, 0) for node in population_nodes))
    
    # Bounds: 0 <= flow <= capacity (reduced by existing traffic)
    bounds = []
    for edge in edges:
        capacity = data_interface.get_property(edge[0], edge[1], 'road_flowrate')
        # Reduce capacity by existing traffic fraction
        available_capacity = capacity * (1 - EXISTING_TRAFFIC)
        bounds.append((0, available_capacity))
    
    # Solve
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Print optimization problem details
    def print_problem_details(c, A_eq, b_eq, bounds, edges, population_nodes, safe_nodes, intermediary_nodes):
        """Print the formal minimum cost flow optimization problem."""
        print("\n=== OPTIMIZATION PROBLEM ===")
        print("Minimize:")
        print(f"  Cost function: sum(c[i] * x[i]) for i in edges")
        print(f"  Number of variables: {len(c)}")
        print("\nSubject to:")
        
        # Print flow conservation constraints
        print("\nFlow Conservation Constraints (Ax = b), measured in cars:")
        for i, (row, b) in enumerate(zip(A_eq, b_eq)):
            if i < len(population_nodes):
                print(f"  Population node {population_nodes[i]}: sum(outflow) - sum(inflow) = {b:.1f} cars")
            elif i < len(population_nodes) + len(intermediary_nodes):
                print(f"  Intermediary node {intermediary_nodes[i-len(population_nodes)]}: sum(outflow) - sum(inflow) = {b:.1f} cars")
            else:
                print(f"  Total flow to safe zones = {b:.1f} cars")
                
        print("\nCapacity Bounds (lb <= x <= ub):")
        print(f"  All flows >= 0")
        print(f"  Available capacity reduced by {EXISTING_TRAFFIC*100:.0f}% for existing traffic")
        for i, ((lb, ub), edge) in enumerate(zip(bounds, edges)):
            if i < 5:  # Print first 5 bounds as example
                print(f"  0 <= flow({edge[0]} → {edge[1]}) <= {ub:.1f}")
            elif i == 5:
                print("  ...")

        print("\nEdge Weights (Cost):")
        for i, edge in enumerate(edges):
            cost = c[i]  # Cost coefficient from objective function
            print(f"  {edge[0]} → {edge[1]}: {cost:.1f}")
                
        print(f"\nNetwork Statistics:")
        print(f"  Population nodes: {len(population_nodes)}")
        print(f"  Safe nodes: {len(safe_nodes)}")
        print(f"  Intermediary nodes: {len(intermediary_nodes)}")
        print(f"  Total edges: {len(edges)}")
        print("\n")
        
    # Print the problem details
    print_problem_details(c, A_eq, b_eq, bounds, edges, population_nodes, safe_nodes, intermediary_nodes)
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        flows = {edges[i]: result.x[i] for i in range(len(edges))}
        return {
            'success': True,
            'flows': flows,
            'total_cost': result.fun,
            'total_flow': sum(flows.values()),
            'population_nodes': population_nodes,
            'safe_nodes': safe_nodes,
            'intermediary_nodes': intermediary_nodes,
            'node_supplies': node_supplies,
            'data_interface': data_interface
        }
    else:
        return {'success': False, 'message': result.message}



def main():
    """Main function to run the minimum cost flow evacuation."""
    # Initialize data interface
    data_interface = DataInterface(
        population_file='data/population.csv',
        road_data_file='data/combined_road_data.csv', 
        adjacency_file='data/expanded_adjacency_matrix.csv'
    )
    
    print(f"Population centers (S): {data_interface.get_population_nodes()}")
    print(f"Safe zones (D): {data_interface.get_safe_nodes()}")
    print(f"Intermediary nodes (T): {data_interface.get_intermediary_nodes()}")
    total_population = data_interface.get_total_population()
    total_cars = total_population / POPULATION_PER_CAR
    print(f"Total population to evacuate: {total_population}")
    print(f"Total car flow required: {total_cars:.1f}")

    # Solve the problem
    result = solve(data_interface)
    
    # Print results
    if result['success']:
        print(f"\n=== SOLUTION ===")
        print(f"Total Cost: {result['total_cost']:.2f}")
        print(f"Total Flow Units (cars): {result['total_flow']:.2f}")
        
        print(f"\nActive Evacuation Routes (flow > 0.1 cars):")
        flows = result['flows']
        active_routes = [(edge, flow) for edge, flow in flows.items() if flow > 0.1]
        active_routes.sort(key=lambda x: x[1], reverse=True)
        
        for edge, flow in active_routes:
            # Get road capacity from data interface
            full_capacity = data_interface.get_property(edge[0], edge[1], 'road_flowrate')
            available_capacity = full_capacity * (1 - EXISTING_TRAFFIC)
            at_capacity = flow >= available_capacity * 0.99  # Allow for small numerical differences
            capacity_str = " (AT CAPACITY)" if at_capacity else ""
            print(f"  {edge[0]} → {edge[1]}: {flow:.1f} cars / {available_capacity:.1f} available capacity (of {full_capacity:.1f} total){capacity_str}")
            
        print(f"\nEdge Scores:")
        for edge, flow in flows.items():
            node1, node2 = edge
            risk = data_interface.get_property(node1, node2, 'road_risk')
            time = data_interface.get_property(node1, node2, 'road_time_minutes')
            distance = data_interface.get_property(node1, node2, 'road_length')
            score = risk * (time + distance) * flow
            print(f"  {node1} → {node2}: {score:.1f}")

        print(f"\nFlow to Safe Zones (cars):")
        for safe_zone in result['safe_nodes']:
            total_inflow = sum(flows.get((node, safe_zone), 0) for node in data_interface.get_all_nodes())
            print(f"  {safe_zone}: {total_inflow:.1f} cars")
            
        # Import and call visualization
        from visualise_optimisation import visualise_all_three
        visualise_all_three(result)
    else:
        print(f"Optimization failed: {result['message']}")
        
    return result


if __name__ == "__main__":
    result = main()
