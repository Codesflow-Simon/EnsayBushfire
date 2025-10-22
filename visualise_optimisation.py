#!/usr/bin/env python3
"""
Visualization for the minimum cost flow optimization results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def getCost(node1: str, node2: str, data_interface) -> float:
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


def visualise_edge_weights(result):
    """
    Visualize the network with edge weights (costs) for optimization.
    
    Args:
        result: Result dictionary from solve function
    """
    if not result['success']:
        print("Cannot visualize - no valid solution")
        return
        
    # Get data from result
    population_nodes = result['population_nodes']
    safe_nodes = result['safe_nodes']
    intermediary_nodes = result['intermediary_nodes']
    data_interface = result['data_interface']
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all nodes
    for node in data_interface.get_all_nodes():
        G.add_node(node)
        
    # Add edges
    for edge in data_interface.get_all_edges():
        if edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(16, 12))
    
    # Create custom layout positioning (same as visualise_network.py)
    pos = {}
    # Alpine region at the top (y=1.5)
    pos['Swifts Creek'] = (0, 1.5)
    pos['Omeo'] = (-0.5, 1.7)
    pos['Benambra'] = (0.5, 1.7)

    # Ensay nodes in the middle (y=0.2 to y=1)
    pos['Doctors'] = (0, 1.0)
    pos['Ensay-Doctors F'] = (-0.3, 0.8)
    pos['Ensay Center'] = (0, 0.6)
    pos['Spring Gulley'] = (0.5, 0.8)
    pos['Ensay North'] = (0.8, 0.6)
    pos['North Bridge'] = (0.2, 0.4)
    pos['Watts Creek'] = (0.8, 0.4)
    pos['Creek Road'] = (0.1, 0.2)
    pos['South Bridge'] = (-0.2, 0.4)

    # Coastal nodes at the bottom (y=-0.2 to y=-1.5)
    pos['Bruthen'] = (0, -0.2)
    pos['Kilmorie Swamp'] = (0.625, -0.4)
    pos['Bairnsdale'] = (-0.625, -1.2)
    pos['Wiseleigh'] = (-0.45, -0.8)
    pos['Nicholson'] = (-0.3, -1.0)
    pos['Tambo Upper'] = (-0.125, -0.8)
    pos['Riversdale Farm'] = (0.45, -0.5)
    pos['Swan Reach'] = (0.6, -1.2)
    pos['Metung'] = (0, -1.5)
    pos['Lakes Entrance'] = (0.875, -1.5)
    
    # Color nodes based on type
    node_colors = []
    for node in G.nodes():
        if node in population_nodes:
            node_colors.append('red')  # Population centers
        elif node in safe_nodes:
            node_colors.append('green')  # Safe zones
        else:
            node_colors.append('lightblue')  # Intermediary nodes
            
    # Prepare edge colors and widths based on cost/weight
    edge_colors = []
    edge_widths = []
    edge_weights = []
    
    # Get all edges from data interface to match the optimization
    all_edges = data_interface.get_all_edges()
    
    for i, edge in enumerate(G.edges()):
        # Get cost for this edge
        try:
            cost = getCost(edge[0], edge[1], data_interface)
        except:
            cost = 0
        
        edge_weights.append(cost)
        
        # Smooth color gradient from grey (low cost) to blue (high cost)
        if cost > 0:
            # Normalize cost to 0-1 range
            max_cost = max(getCost(e[0], e[1], data_interface) for e in all_edges)
            intensity = min(cost / max_cost, 1.0)
            
            # Continuous interpolation from grey to dark blue
            # Grey: (0.8, 0.8, 0.8) -> Dark Blue: (0.1, 0.2, 0.9)
            r = 0.8 - intensity * 0.7  # 0.8 -> 0.1
            g = 0.8 - intensity * 0.6  # 0.8 -> 0.2  
            b = 0.8 + intensity * 0.1  # 0.8 -> 0.9
            
            # Ensure values stay in valid range
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            edge_colors.append((r, g, b, 0.8))  # RGBA with good alpha
            edge_widths.append(max(1, min(cost / 10, 8)))  # Width based on cost
        else:
            edge_colors.append((0.8, 0.8, 0.8, 0.5))  # Light gray for no cost
            edge_widths.append(0.5)
    
    # Draw network with cost-based styling
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8)
    
    # Add edge labels for cost values on ALL edges
    edge_labels = {}
    for i, edge in enumerate(G.edges()):
        cost = edge_weights[i]
        edge_labels[edge] = f"{cost:.1f}"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, font_color='darkblue')
                                 
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Population Centers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Safe Zones'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Intermediary Nodes'),
        plt.Line2D([0], [0], color=(0.05, 0.3, 0.95), linewidth=4, label='High Cost (dark blue)'),
        plt.Line2D([0], [0], color=(0.1, 0.5, 0.9), linewidth=3, label='Medium Cost (blue)'),
        plt.Line2D([0], [0], color=(0.2, 0.6, 0.8), linewidth=2, label='Low Cost (light blue)'),
        plt.Line2D([0], [0], color=(0.6, 0.6, 0.6), linewidth=1, label='Very Low Cost (grey)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title(f'Edge Weights (Costs) for Optimization\n'
             f'Cost = Risk × (Time + Distance)', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualise_optimisation(result):
    """
    Visualize the network with flow solution.
    
    Args:
        result: Result dictionary from solve function
    """
    if not result['success']:
        print("Cannot visualize - no valid solution")
        return
        
    # Get data from result
    flows = result['flows']
    population_nodes = result['population_nodes']
    safe_nodes = result['safe_nodes']
    intermediary_nodes = result['intermediary_nodes']
    data_interface = result['data_interface']
    
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all nodes
    for node in data_interface.get_all_nodes():
        G.add_node(node)
        
    # Add edges
    for edge in data_interface.get_all_edges():
        if edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(edge[0], edge[1])
    
    plt.figure(figsize=(16, 12))
    
    # Create custom layout positioning (same as visualise_network.py)
    pos = {}
    # Alpine region at the top (y=1.5)
    pos['Swifts Creek'] = (0, 1.5)
    pos['Omeo'] = (-0.5, 1.7)
    pos['Benambra'] = (0.5, 1.7)

    # Ensay nodes in the middle (y=0.2 to y=1)
    pos['Doctors'] = (0, 1.0)
    pos['Ensay-Doctors F'] = (-0.3, 0.8)
    pos['Ensay Center'] = (0, 0.6)
    pos['Spring Gulley'] = (0.5, 0.8)
    pos['Ensay North'] = (0.8, 0.6)
    pos['North Bridge'] = (0.2, 0.4)
    pos['Watts Creek'] = (0.8, 0.4)
    pos['Creek Road'] = (0.1, 0.2)
    pos['South Bridge'] = (-0.2, 0.4)

    # Coastal nodes at the bottom (y=-0.2 to y=-1.5)
    pos['Bruthen'] = (0, -0.2)
    pos['Kilmorie Swamp'] = (0.625, -0.4)
    pos['Bairnsdale'] = (-0.625, -1.2)
    pos['Wiseleigh'] = (-0.45, -0.8)
    pos['Nicholson'] = (-0.3, -1.0)
    pos['Tambo Upper'] = (-0.125, -0.8)
    pos['Riversdale Farm'] = (0.45, -0.5)
    pos['Swan Reach'] = (0.6, -1.2)
    pos['Metung'] = (0, -1.5)
    pos['Lakes Entrance'] = (0.875, -1.5)

    
    # Color nodes based on type
    node_colors = []
    for node in G.nodes():
        if node in population_nodes:
            node_colors.append('red')  # Population centers
        elif node in safe_nodes:
            node_colors.append('green')  # Safe zones
        else:
            node_colors.append('lightblue')  # Intermediary nodes
            
    # Prepare edge colors and widths based on flow
    edge_colors = []
    edge_widths = []
    
    # Get all edges from data interface to match the optimization
    all_edges = data_interface.get_all_edges()
    
    for i, edge in enumerate(G.edges()):
        # For undirected graph, we need to sum flows from both directions
        flow_forward = flows.get(edge, 0)
        flow_reverse = flows.get((edge[1], edge[0]), 0)
        total_flow = abs(flow_forward) + abs(flow_reverse)  # Use absolute values
        
        # Smooth color gradient from gray (0 flow) to red (max flow)
        if total_flow > 0:
            # Normalize flow to 0-1 range
            max_flow = max(abs(f) for f in flows.values()) if flows else 1
            intensity = min(total_flow / max_flow, 1.0)
            
            # Continuous interpolation from grey to bright red
            # Grey: (0.7, 0.7, 0.7) -> Bright Red: (1.0, 0.1, 0.1)
            r = 0.7 + intensity * 0.3  # 0.7 -> 1.0
            g = 0.7 - intensity * 0.6  # 0.7 -> 0.1
            b = 0.7 - intensity * 0.6  # 0.7 -> 0.1
            
            # Ensure values stay in valid range
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            edge_colors.append((r, g, b, 0.8))  # RGBA with good alpha
            edge_widths.append(max(1, min(total_flow / 5, 8)))  # Width based on flow
        else:
            edge_colors.append((0.8, 0.8, 0.8, 0.5))  # Light gray for no flow
            edge_widths.append(0.5)
    
    # Draw network with flow-based styling
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8)
    
    # Add edge labels for flow values on ALL edges
    edge_labels = {}
    for i, edge in enumerate(G.edges()):
        # For undirected graph, we need to sum flows from both directions
        flow_forward = flows.get(edge, 0)
        flow_reverse = flows.get((edge[1], edge[0]), 0)
        total_flow = abs(flow_forward) + abs(flow_reverse)  # Use absolute values
        
        if total_flow > 0:
            edge_labels[edge] = f"{total_flow:.1f}"
        else:
            edge_labels[edge] = "0.0"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, font_color='darkred')
                                 
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Population Centers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=10, label='Safe Zones'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Intermediary Nodes'),
        plt.Line2D([0], [0], color=(1.0, 0.1, 0.05), linewidth=4, label='High Flow (bright red)'),
        plt.Line2D([0], [0], color=(0.8, 0.2, 0.1), linewidth=3, label='Medium Flow (orange-red)'),
        plt.Line2D([0], [0], color=(0.6, 0.4, 0.3), linewidth=2, label='Low Flow (grayish red)'),
        plt.Line2D([0], [0], color=(0.8, 0.8, 0.8), linewidth=1, label='No Flow (gray)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.title(f'Minimum Cost Flow Evacuation Solution\n'
             f'Total Cost: {result["total_cost"]:.2f}, '
             f'Population per Car: 1.5', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def visualise_all_three(result):
    """
    Visualize edge weights, flow solution, and capacity as three separate figures.
    
    Args:
        result: Result dictionary from solve function
    """
    if not result['success']:
        print("Cannot visualize - no valid solution")
        return
        
    # Get data from result
    flows = result['flows']
    population_nodes = result['population_nodes']
    safe_nodes = result['safe_nodes']
    intermediary_nodes = result['intermediary_nodes']
    data_interface = result['data_interface']
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all nodes
    for node in data_interface.get_all_nodes():
        G.add_node(node)
        
    # Add edges
    for edge in data_interface.get_all_edges():
        if edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(edge[0], edge[1])
    
    # Create custom layout positioning
    pos = {}
    # Alpine region at the top (y=1.5)
    pos['Swifts Creek'] = (0, 1.5)
    pos['Omeo'] = (-0.5, 1.7)
    pos['Benambra'] = (0.5, 1.7)

    # Ensay nodes in the middle (y=0.2 to y=1)
    pos['Doctors'] = (0, 1.0)
    pos['Ensay-Doctors F'] = (-0.3, 0.8)
    pos['Ensay Center'] = (0, 0.6)
    pos['Spring Gulley'] = (0.5, 0.8)
    pos['Ensay North'] = (0.8, 0.6)
    pos['North Bridge'] = (0.2, 0.4)
    pos['Watts Creek'] = (0.8, 0.4)
    pos['Creek Road'] = (0.1, 0.2)
    pos['South Bridge'] = (-0.2, 0.4)

    # Coastal nodes at the bottom (y=-0.2 to y=-1.5)
    pos['Bruthen'] = (0, -0.2)
    pos['Kilmorie Swamp'] = (0.625, -0.4)
    pos['Bairnsdale'] = (-0.625, -1.2)
    pos['Wiseleigh'] = (-0.45, -0.8)
    pos['Nicholson'] = (-0.3, -1.0)
    pos['Tambo Upper'] = (-0.125, -0.8)
    pos['Riversdale Farm'] = (0.45, -0.5)
    pos['Swan Reach'] = (0.6, -1.2)
    pos['Metung'] = (0, -1.5)
    pos['Lakes Entrance'] = (0.875, -1.5)
    
    # Color nodes based on type
    node_colors = []
    for node in G.nodes():
        if node in population_nodes:
            node_colors.append('red')  # Population centers
        elif node in safe_nodes:
            node_colors.append('green')  # Safe zones
        else:
            node_colors.append('lightblue')  # Intermediary nodes
    
    # Create three separate figures
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
    
    # === LEFT PLOT: Edge Weights (Costs) ===
    # Prepare edge colors and widths based on cost/weight
    edge_colors_weights = []
    edge_widths_weights = []
    edge_weights = []
    
    # Get all edges from data interface to match the optimization
    all_edges = data_interface.get_all_edges()
    
    for i, edge in enumerate(G.edges()):
        # Get cost for this edge
        try:
            cost = getCost(edge[0], edge[1], data_interface)
        except:
            cost = 0
        
        edge_weights.append(cost)
        
        # Smooth color gradient from grey (low cost) to blue (high cost)
        if cost > 0:
            # Normalize cost to 0-1 range
            max_cost = max(getCost(e[0], e[1], data_interface) for e in all_edges)
            intensity = min(cost / max_cost, 1.0)
            
            # Continuous interpolation from grey to dark blue
            # Grey: (0.8, 0.8, 0.8) -> Dark Blue: (0.1, 0.2, 0.9)
            r = 0.8 - intensity * 0.7  # 0.8 -> 0.1
            g = 0.8 - intensity * 0.6  # 0.8 -> 0.2  
            b = 0.8 + intensity * 0.1  # 0.8 -> 0.9
            
            # Ensure values stay in valid range
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            edge_colors_weights.append((r, g, b, 0.8))  # RGBA with good alpha
            edge_widths_weights.append(max(1, min(cost / 10, 8)))  # Width based on cost
        else:
            edge_colors_weights.append((0.8, 0.8, 0.8, 0.5))  # Light gray for no cost
            edge_widths_weights.append(0.5)
    
    # Draw network with cost-based styling
    nx.draw(G, pos, ax=ax1,
            node_color=node_colors,
            node_size=800,
            font_size=7,
            font_weight='bold',
            with_labels=True,
            edge_color=edge_colors_weights,
            width=edge_widths_weights,
            alpha=0.8)
    
    # Add edge labels for cost values
    edge_labels_weights = {}
    for i, edge in enumerate(G.edges()):
        cost = edge_weights[i]
        edge_labels_weights[edge] = f"{cost:.1f}"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels_weights, ax=ax1, font_size=5, font_color='darkblue')
    
    # Add legend for weights
    legend_elements_weights = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=8, label='Population Centers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=8, label='Safe Zones'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=8, label='Intermediary Nodes'),
        plt.Line2D([0], [0], color=(0.2, 0.9, 1.0), linewidth=3, label='High Cost (dark blue)'),
        plt.Line2D([0], [0], color=(0.3, 0.8, 0.9), linewidth=2, label='Medium Cost (blue)'),
        plt.Line2D([0], [0], color=(0.4, 0.6, 0.8), linewidth=1.5, label='Low Cost (light blue)'),
        plt.Line2D([0], [0], color=(0.6, 0.6, 0.6), linewidth=1, label='Very Low Cost (grey)')
    ]
    ax1.legend(handles=legend_elements_weights, loc='upper left', fontsize=8)
    ax1.set_title(f'Edge Weights (Costs) for Optimization\n'
                 f'Cost = Risk × (Time + Distance)', 
                 fontsize=12, fontweight='bold')
    
    # === RIGHT PLOT: Flow Solution ===
    # Prepare edge colors and widths based on flow
    edge_colors_flow = []
    edge_widths_flow = []
    
    for i, edge in enumerate(G.edges()):
        # For undirected graph, we need to sum flows from both directions
        flow_forward = flows.get(edge, 0)
        flow_reverse = flows.get((edge[1], edge[0]), 0)
        total_flow = abs(flow_forward) + abs(flow_reverse)  # Use absolute values
        
        # Smooth color gradient from gray (0 flow) to red (max flow)
        if total_flow > 0:
            # Normalize flow to 0-1 range
            max_flow = max(abs(f) for f in flows.values()) if flows else 1
            intensity = min(total_flow / max_flow, 1.0)
            
            # Continuous interpolation from grey to bright red
            # Grey: (0.7, 0.7, 0.7) -> Bright Red: (1.0, 0.1, 0.1)
            r = 0.7 + intensity * 0.3  # 0.7 -> 1.0
            g = 0.7 - intensity * 0.6  # 0.7 -> 0.1
            b = 0.7 - intensity * 0.6  # 0.7 -> 0.1
            
            # Ensure values stay in valid range
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            edge_colors_flow.append((r, g, b, 0.8))  # RGBA with good alpha
            edge_widths_flow.append(max(1, min(total_flow / 5, 8)))  # Width based on flow
        else:
            edge_colors_flow.append((0.8, 0.8, 0.8, 0.5))  # Light gray for no flow
            edge_widths_flow.append(0.5)
    
    # Draw network with flow-based styling
    nx.draw(G, pos, ax=ax2,
            node_color=node_colors,
            node_size=800,
            font_size=7,
            font_weight='bold',
            with_labels=True,
            edge_color=edge_colors_flow,
            width=edge_widths_flow,
            alpha=0.8)
    
    # Add edge labels for flow values
    edge_labels_flow = {}
    for i, edge in enumerate(G.edges()):
        # For undirected graph, we need to sum flows from both directions
        flow_forward = flows.get(edge, 0)
        flow_reverse = flows.get((edge[1], edge[0]), 0)
        total_flow = abs(flow_forward) + abs(flow_reverse)  # Use absolute values
        
        if total_flow > 0:
            edge_labels_flow[edge] = f"{total_flow:.1f}"
        else:
            edge_labels_flow[edge] = "0.0"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels_flow, ax=ax2, font_size=5, font_color='darkred')
    
    # Add legend for flow
    legend_elements_flow = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=8, label='Population Centers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=8, label='Safe Zones'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=8, label='Intermediary Nodes'),
        plt.Line2D([0], [0], color=(1.0, 0.1, 0.05), linewidth=3, label='High Flow (bright red)'),
        plt.Line2D([0], [0], color=(0.8, 0.2, 0.1), linewidth=2, label='Medium Flow (orange-red)'),
        plt.Line2D([0], [0], color=(0.6, 0.4, 0.3), linewidth=1.5, label='Low Flow (grayish red)'),
        plt.Line2D([0], [0], color=(0.8, 0.8, 0.8), linewidth=1, label='No Flow (gray)')
    ]
    ax2.legend(handles=legend_elements_flow, loc='upper left', fontsize=8)
    ax2.set_title(f'Minimum Cost Flow Evacuation Solution\n'
                 f'Total Cost: {result["total_cost"]:.2f}, '
                 f'Total Flow: {result["total_flow"]:.1f} cars', 
                 fontsize=12, fontweight='bold')
    
    # === FIGURE 3: Road Capacity ===
    # Prepare edge colors and widths based on capacity
    edge_colors_capacity = []
    edge_widths_capacity = []
    edge_capacities = []
    
    for i, edge in enumerate(G.edges()):
        # Get capacity for this edge
        try:
            capacity = data_interface.get_property(edge[0], edge[1], 'road_flowrate')
        except:
            capacity = 0
        
        edge_capacities.append(capacity)
        
        if capacity > 0:
            # Normalize capacity to 0-1 range
            max_capacity = max(data_interface.get_property(e[0], e[1], 'road_flowrate') for e in all_edges)
            intensity = min(capacity / max_capacity, 1.0)
            
            # Continuous interpolation from grey to green
            # Grey: (0.8, 0.8, 0.8) -> Green: (0.1, 0.8, 0.1)
            r = 0.8 - intensity * 0.7  # 0.8 -> 0.1
            g = 0.8  # Keep green constant
            b = 0.8 - intensity * 0.7  # 0.8 -> 0.1
            
            # Ensure values stay in valid range
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            
            edge_colors_capacity.append((r, g, b, 0.8))  # RGBA with good alpha
            edge_widths_capacity.append(max(1, min(capacity / 50, 8)))  # Width based on capacity
        else:
            edge_colors_capacity.append((0.8, 0.8, 0.8, 0.5))  # Light gray for no capacity
            edge_widths_capacity.append(0.5)
    
    # Draw network with capacity-based styling
    nx.draw(G, pos, ax=ax3,
            node_color=node_colors,
            node_size=800,
            font_size=7,
            font_weight='bold',
            with_labels=True,
            edge_color=edge_colors_capacity,
            width=edge_widths_capacity,
            alpha=0.8)
    
    # Add edge labels for capacity values
    edge_labels_capacity = {}
    for i, edge in enumerate(G.edges()):
        capacity = edge_capacities[i]
        edge_labels_capacity[edge] = f"{capacity:.0f}"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels_capacity, ax=ax3, font_size=5, font_color='darkgreen')
    
    # Add legend for capacity
    legend_elements_capacity = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=8, label='Population Centers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                  markersize=8, label='Safe Zones'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=8, label='Intermediary Nodes'),
        plt.Line2D([0], [0], color=(0.1, 0.8, 0.1), linewidth=3, label='High Capacity (dark green)'),
        plt.Line2D([0], [0], color=(0.4, 0.8, 0.4), linewidth=2, label='Medium Capacity (green)'),
        plt.Line2D([0], [0], color=(0.6, 0.8, 0.6), linewidth=1.5, label='Low Capacity (light green)'),
        plt.Line2D([0], [0], color=(0.8, 0.8, 0.8), linewidth=1, label='Very Low Capacity (grey)')
    ]
    ax3.legend(handles=legend_elements_capacity, loc='upper left', fontsize=8)
    ax3.set_title(f'Road Capacity (Flow Rate)\n'
                 f'Maximum vehicles per hour', 
                 fontsize=12, fontweight='bold')
    
    # Show all three figures
    plt.show()


if __name__ == "__main__":
    # For testing purposes
    print("This module should be imported and used with a result dictionary from solve()")