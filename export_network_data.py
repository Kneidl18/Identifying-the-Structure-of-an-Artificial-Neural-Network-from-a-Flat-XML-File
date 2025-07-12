#!/usr/bin/env python3
"""
Comprehensive Neural Network Data Export Tool

This script provides functionality to export all available neural network data
from XNET files to comprehensive text reports. It extracts and formats all
information accessible through the current codebase.

Usage:
    python export_network_data.py <path_to_xnet_file> [output_file]

Examples:
    python export_network_data.py network.xnet
    python export_network_data.py network.xnet custom_report.txt
    python export_network_data.py xnet_files/star4_24_12.xnet
"""

import sys
import os
from pathlib import Path
from xnet_parser import Network


def export_network_data(xnet_path: str, output_path: str = None) -> str:
    """
    Export comprehensive neural network data to a text file.
    
    Args:
        xnet_path: Path to the XNET file to analyze
        output_path: Optional custom output path. If None, generates from input filename.
        
    Returns:
        Path to the generated output file
        
    Raises:
        FileNotFoundError: If the XNET file doesn't exist
        RuntimeError: If parsing fails
    """
    # Validate input file
    if not os.path.exists(xnet_path):
        raise FileNotFoundError(f"XNET file not found: {xnet_path}")
    
    print(f"Loading neural network from: {xnet_path}")
    
    try:
        # Load and parse the network
        network = Network(xnet_path)
        print(f"Successfully loaded network with {network.count_layers()} layers and {network.count_neurons():,} neurons")
        
        # Generate output path if not provided
        if output_path is None:
            base_name = Path(xnet_path).stem
            output_path = f"{base_name}.txt"
        
        print(f"Exporting comprehensive data to: {output_path}")
        
        # Export all data
        result_path = network.export_comprehensive_data(output_path)
        
        # Get file size for confirmation
        file_size = os.path.getsize(result_path)
        print(f"Export completed successfully!")
        print(f"Output file: {result_path}")
        print(f"File size: {file_size:,} bytes")
        
        return result_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to export network data: {e}")


def main():
    """Main entry point for the export tool."""
    if len(sys.argv) < 2:
        print("Error: Missing required argument")
        print()
        print("Usage: python export_network_data.py <path_to_xnet_file> [output_file]")
        print()
        print("Examples:")
        print("  python export_network_data.py network.xnet")
        print("  python export_network_data.py network.xnet custom_report.txt")
        print("  python export_network_data.py xnet_files/star4_24_12.xnet")
        sys.exit(1)
    
    xnet_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    

if __name__ == "__main__":
    main()
