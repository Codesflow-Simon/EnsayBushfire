#!/usr/bin/env python3
"""
Concise script to visualize the node network from adjacency matrix
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read adjacency matrix
df = pd.read_csv('data/expanded_adjacency_matrix.csv', index_col=0)

# Create network graph
G = nx.Graph()

# Add nodes
for node in df.index:
    G.add_node(node)

# Add edges with weights
for i, source in enumerate(df.index):
    for j, target in enumerate(df.columns):
        weight = df.iloc[i, j]
        if weight > 0:
            G.add_edge(source, target, weight=weight)

# Set up the plot
plt.figure(figsize=(16, 12))

# Define node regions
ensay_nodes = ['Doctors', 'Ensay-Doctors F', 'Ensay Center', 'Spring Gulley', 'Ensay North', 
               'North Bridge', 'Watts Creek', 'Creek Road', 'South Bridge']
alpine_nodes = ['Swifts Creek', 'Omeo', 'Benambra']
coastal_nodes = ['Bairnsdale', 'Wiseleigh', 'Nicholson', 'Tambo Upper', 'Kilmorie Swamp',
                 'Riversdale Farm', 'Swan Reach', 'Metung', 'Lakes Entrance', 'Bruthen']

# Create custom layout positioning
pos = {}
# Alpine region at the top (y=2)
pos['Swifts Creek'] = (0, 2)
pos['Omeo'] = (-0.5, 2.2)
pos['Benambra'] = (0.5, 2.2)

# Ensay nodes in the middle (y=0 to y=1)
pos['Doctors'] = (0, 1)
pos['Ensay-Doctors F'] = (-0.3, 0.8)
pos['Ensay Center'] = (0, 0.5)
pos['Spring Gulley'] = (0.5, 0.8)
pos['Ensay North'] = (0.8, 0.5)
pos['North Bridge'] = (0.2, 0.2)
pos['Watts Creek'] = (0.8, 0.2)
pos['Creek Road'] = (0.1, 0)
pos['South Bridge'] = (-0.2, 0.2)

# Coastal nodes at the bottom (y=-1 to y=-2.5)
# Bruthen as central hub connecting to Kilmorie Swamp
pos['Bruthen'] = (0, -0.5)
pos['Kilmorie Swamp'] = (0.625, -0.7)
pos['Bairnsdale'] = (-0.625, -1.1) 
pos['Wiseleigh'] = (-0.45, -0.9)
pos['Nicholson'] = (-0.3, -1.0)
pos['Tambo Upper'] = (-0.125, -0.9)
pos['Riversdale Farm'] = (0.45, -0.75)
pos['Swan Reach'] = (0.6, -1.1)
pos['Metung'] = (0, -1.25)
pos['Lakes Entrance'] = (0.875, -1.25)

node_colors = []
for node in G.nodes():
    if node in ensay_nodes:
        node_colors.append('lightblue')
    elif node in alpine_nodes:
        node_colors.append('lightgreen')
    elif node in coastal_nodes:
        node_colors.append('lightcoral')
    else:
        node_colors.append('lightgray')

# Draw the network
nx.draw(G, pos, 
        node_color=node_colors,
        node_size=1000,
        font_size=8,
        font_weight='bold',
        with_labels=True,
        edge_color='gray',
        width=1,
        alpha=0.8)

# Add edge labels for weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Alpine Region (Top)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Ensay Region (Middle)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Coastal Region (Bottom)')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

plt.title('Ensay Bushfire Evacuation Network', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print network statistics
print(f"Network Statistics:")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
