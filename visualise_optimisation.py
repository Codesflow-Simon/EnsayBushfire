#!/usr/bin/env python3
"""
Visualization for the minimum cost flow optimization results.
"""

import networkx as nx
import matplotlib.pyplot as plt


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
    # Bruthen as central hub connecting to Kilmorie Swamp
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
            
            # Create smooth gradient: gray -> light red -> bright red
            if intensity < 0.1:  # Very low flow - light gray
                r, g, b = 0.7, 0.7, 0.7
            elif intensity < 0.3:  # Low flow - grayish red
                r = 0.5 + intensity * 0.5
                g = 0.5 - intensity * 0.2
                b = 0.5 - intensity * 0.2
            elif intensity < 0.6:  # Medium flow - orange-red
                r = 0.8 + intensity * 0.2
                g = 0.4 - intensity * 0.2
                b = 0.3 - intensity * 0.2
            else:  # High flow - bright red
                r = 1.0
                g = 0.2 - intensity * 0.1
                b = 0.1 - intensity * 0.05
            
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


if __name__ == "__main__":
    # For testing purposes
    print("This module should be imported and used with a result dictionary from solve()")