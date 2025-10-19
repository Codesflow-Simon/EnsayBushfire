#!/usr/bin/env python3
"""
DataInterface class for managing bushfire evacuation data.
Provides clean interface for accessing population, adjacency, and road properties.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class DataInterface:
    """
    Interface for accessing bushfire evacuation data including population,
    adjacency relationships, and road properties.
    """
    
    def __init__(self, population_file: str, road_data_file: str, adjacency_file: str):
        """
        Initialize DataInterface with data files.
        
        Args:
            population_file: Path to population data CSV
            road_data_file: Path to road data CSV with flowrates
            adjacency_file: Path to adjacency matrix CSV
        """
        # Load data
        self.population_df = pd.read_csv(population_file)
        self.road_df = pd.read_csv(road_data_file)
        self.adjacency_df = pd.read_csv(adjacency_file, index_col=0)
        
        # Create node name mapping to handle inconsistencies
        self.node_mapping = {
            'Spring Gulley Intersection': 'Spring Gulley',
            'North Bridge': 'North Bridge',
            'Creek Road Intersection': 'Creek Road',
            'Doctors Flat': 'Doctors',
            'Ensay-Doctors Road': 'Ensay-Doctors F',
            'Ensay Centre': 'Ensay Center',
            'Spring Gully': 'Spring Gulley'
        }
        
        # Create reverse mapping
        self.reverse_mapping = {v: k for k, v in self.node_mapping.items()}
        
        # Process data
        self._process_population()
        self._process_roads()
        
    def _process_population(self):
        """Process population data with node name mapping."""
        self.population_data = {}
        
        for _, row in self.population_df.iterrows():
            # Map node names to consistent format
            mapped_name = self.node_mapping.get(row['node'], row['node'])
            self.population_data[mapped_name] = row['population']
            
    def _process_roads(self):
        """Process road data to create edge lookup."""
        self.road_properties = {}
        
        for _, row in self.road_df.iterrows():
            if pd.notna(row['node_1']) and pd.notna(row['node_2']):
                # Map node names to consistent format
                node1 = self.node_mapping.get(row['node_1'], row['node_1'])
                node2 = self.node_mapping.get(row['node_2'], row['node_2'])
                
                # Create edge key (order doesn't matter for undirected graph)
                edge_key = tuple(sorted([node1, node2]))
                
                # Store all road properties
                self.road_properties[edge_key] = {
                    'road_name': row['road_name'],
                    'road_length': row['road_length'],
                    'road_time_minutes': row['road_time_minutes'],
                    'road_speed': row['road_speed'],
                    'road_flowrate': row['road_flowrate'],
                    'road_risk': row['road_risk'],
                    'region': row['region']
                }
                
    def get_population(self, node: str) -> Union[float, int]:
        """
        Get population for a given node.
        
        Args:
            node: Node name as string
            
        Returns:
            Population value (positive for population centers, negative for safe zones, 0 for others)
            Returns negative infinity if node not found
        """
        # Map node name if needed
        mapped_node = self.node_mapping.get(node, node)
        
        if mapped_node in self.population_data:
            return self.population_data[mapped_node]
        else:
            return float('-inf')
            
    def get_adjacency(self, node1: str, node2: str) -> bool:
        """
        Check if two nodes are adjacent (connected by a road).
        
        Args:
            node1: First node name
            node2: Second node name
            
        Returns:
            True if nodes are connected, False otherwise
        """
        # Map node names if needed
        mapped_node1 = self.node_mapping.get(node1, node1)
        mapped_node2 = self.node_mapping.get(node2, node2)
        
        # Create edge key
        edge_key = tuple(sorted([mapped_node1, mapped_node2]))
        
        return edge_key in self.road_properties
        
    def get_property(self, node1: str, node2: str, property_name: str) -> float:
        """
        Get a specific property for the road between two nodes.
        
        Args:
            node1: First node name
            node2: Second node name
            property_name: Name of the property (e.g., 'road_flowrate', 'road_risk')
            
        Returns:
            Property value
            
        Raises:
            ValueError: If nodes are not connected or property doesn't exist
        """
        # Map node names if needed
        mapped_node1 = self.node_mapping.get(node1, node1)
        mapped_node2 = self.node_mapping.get(node2, node2)
        
        # Create edge key
        edge_key = tuple(sorted([mapped_node1, mapped_node2]))
        
        if edge_key not in self.road_properties:
            raise ValueError(f"Nodes '{node1}' and '{node2}' are not connected")
            
        if property_name not in self.road_properties[edge_key]:
            raise ValueError(f"Property '{property_name}' not found for road between '{node1}' and '{node2}'")
            
        return self.road_properties[edge_key][property_name]
        
    def get_all_nodes(self) -> list:
        """Get list of all nodes in the network."""
        return list(self.adjacency_df.index)
        
    def get_population_nodes(self) -> list:
        """Get list of nodes with positive population."""
        return [node for node, pop in self.population_data.items() if pop > 0]
        
    def get_safe_nodes(self) -> list:
        """Get list of nodes with negative population (safe zones)."""
        return [node for node, pop in self.population_data.items() if pop < 0]
        
    def get_intermediary_nodes(self) -> list:
        """Get list of intermediary nodes (zero population)."""
        all_nodes = set(self.get_all_nodes())
        population_safe_nodes = set(self.get_population_nodes() + self.get_safe_nodes())
        return list(all_nodes - population_safe_nodes)
        
    def get_total_population(self) -> float:
        """Get total population to evacuate."""
        return sum(pop for pop in self.population_data.values() if pop > 0)
        
    def get_all_edges(self) -> list:
        """Get list of all edges in the network."""
        return [(node1, node2) for node1 in self.get_all_nodes() 
                for node2 in self.get_all_nodes() 
                if self.get_adjacency(node1, node2)]
