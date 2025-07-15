# Neural Network Analysis Tool

A Python tool for analyzing neural networks from XNET files, with special focus on convolutional neural network analysis and padding detection.

## Overview

This project provides analysis capabilities for neural networks stored in XNET format (compressed XML). The tool can extract network structure, analyze convolutional layers, detect padding parameters, and provide detailed layer information with comprehensive statistics.

## Features

- **XNET File Parsing**: Parse compressed XML neural network files
- **Layer Analysis**: Detailed analysis of network layers and connections
- **Convolutional Layer Detection**: Automatic identification of convolutional layers
- **Padding Analysis**: Detection and analysis of convolutional layer padding
- **Map Analysis**: Analysis of feature maps in convolutional layers
- **Network Statistics**: Comprehensive network structure statistics
- **MultiLink Analysis**: Specialized analysis of MultiLink connections for convolution operations
- **Comprehensive Data Export**: Export all network data to structured text files instead of terminal output

## Installation

### Prerequisites

- Python 3.11 or higher
- Standard Python libraries (gzip, xml.etree.ElementTree, math, sys)

### Setup

1. Clone or download this repository
2. No additional dependencies required - uses only Python standard library

## Usage

### Basic Usage

```python
from xnet_parser import Network

# Load a network from XNET file
network = Network("path/to/your/network.xnet")

# Get basic network information
print(network.generic_info())

# Get detailed layer analysis with padding information
print(network.full_layer_info(show_padding=True))

# Get layer analysis without padding
print(network.full_layer_info(show_padding=False))

# Get convolutional layer maps information
print(network.convolutional_layer_maps_info())

# Export comprehensive data to text file
output_path = network.export_comprehensive_data()
print(f"Data exported to: {output_path}")

# Export with custom filename
output_path = network.export_comprehensive_data("custom_report.txt")
```

### Running the Main Analysis

```bash
cd code
# Print analysis to terminal (original behavior)
python3 main.py path/to/your/network.xnet

# Export comprehensive data to text file
python3 main.py path/to/your/network.xnet --export
```

This will run a comprehensive analysis of the specified network file, displaying detailed layer information including padding analysis for convolutional layers.

### Comprehensive Data Export

```bash
# Using the dedicated export script
python3 export_network_data.py path/to/your/network.xnet

# Export with custom output filename
python3 export_network_data.py network.xnet custom_report.txt

# Using sample files
python3 export_network_data.py xnet_files/star4_24_12.xnet
```

The export functionality creates comprehensive text files containing ALL accessible network data:
- Complete network structure and statistics
- Detailed layer information and properties
- Comprehensive neuron data and connectivity
- Link and MultiLink analysis
- Map information with padding analysis
- Network topology and connectivity patterns

**Note**: Both automatic filename generation (based on input file) and custom output filenames are fully supported and working correctly.

### Command Line Usage

The tool accepts command line arguments for both analysis modes:

```bash
# Terminal output
python3 main.py xnet_files/star4_24_12.xnet

# File export
python3 main.py xnet_files/star4_24_12.xnet --export
python3 export_network_data.py xnet_files/star4_24_12.xnet
```

## File Structure

```
.
├── code/
│   ├── README.md            # This file
│   ├── main.py              # Main analysis script
│   ├── export_network_data.py # Comprehensive data export script
│   ├── xnet_parser.py       # XNET file parser and Network class
│   ├── util.py              # Utility functions
│   ├── components/          # Core data structures
│   │   ├── __init__.py
│   │   ├── layer.py         # Layer and Map classes
│   │   ├── link.py          # Link and MultiLinkBlock classes
│   │   └── neuron.py        # Neuron class
│   ├── xnet_files/          # Sample XNET files
│   │   ├── star3_208_54.xnet
│   │   ├── star3_48_25.xnet
│   │   ├── star4_24_12.xnet
│   │   └── star7_109_15.xnet
│   ├── methods.txt          # Analysis method documentation
│   └── multilink_blocks.txt # MultiLink analysis documentation
├── Report/                  # Project documentation and reports
├── vortrag_*/              # Presentation materials
└── xml/                    # Temporary XML files (automatically cleaned up)
```

## XNET File Format

XNET files are gzip-compressed XML files containing neural network definitions. The format includes:

- **Layers**: Network layer definitions with metadata
- **Neurons**: Individual neuron specifications with positions and properties
- **Links**: Connections between neurons (regular Links and MultiLinks)
- **Maps**: Feature map definitions for convolutional layers

### MultiLinks

MultiLinks are special connections used in convolutional layers where multiple connections share the same ID, representing kernel operations applied across different spatial positions.

## API Reference

### Core Classes

#### `Network`
Main class for loading and analyzing neural networks.

```python
network = Network("path/to/file.xnet")

# Export comprehensive data
output_path = network.export_comprehensive_data()
```

**Key Methods:**
- `export_comprehensive_data(output_path=None)` - Export all network data to text file
- `generic_info()` - Returns basic network statistics
- `full_layer_info(show_padding=True)` - Returns detailed layer analysis
- `convolutional_layer_maps_info()` - Returns convolutional layer map information
- `count_layers()` - Returns number of layers
- `count_neurons()` - Returns total number of neurons
- `count_links()` - Returns number of regular links
- `count_multilinks()` - Returns number of MultiLinks
- `count_unique_multilinks()` - Returns number of unique MultiLink IDs
- `count_anylink()` - Returns total number of connections (links + multilinks)
- `count_connections_from_layer(layer_level)` - Returns link counts for specific layer

#### `Neuron`
Represents individual neurons with properties like ID, layer assignment, and connections.

**Attributes:**
- `id` - Unique neuron identifier
- `layer` - Layer level assignment
- `links_in` - Dictionary of incoming connections
- `links_out` - Dictionary of outgoing connections

#### `Layer`
Represents network layers with collections of neurons and metadata.

**Attributes:**
- `level` - Layer level in network hierarchy
- `neurons` - List of neurons in this layer
- `maps` - List of feature maps (for convolutional layers)
- `is_convolutional` - Boolean indicating if layer is convolutional

#### `Map`
Represents feature maps in convolutional layers.

**Attributes:**
- `layer_level` - Associated layer level
- `size` - Tuple representing map dimensions
- `conn_num` - Connection number (kernel size indicator)
- `padding` - Tuple of padding values (top, right, bottom, left)

## Examples

### Example 1: Basic Network Analysis

```python
from xnet_parser import Network

# Load network
network = Network("xnet_files/star4_24_12.xnet")

# Get basic network statistics
print(network.generic_info())

# Get detailed layer information
print(network.full_layer_info(show_padding=True))

# Get basic counts
print(f"Network has {network.count_layers()} layers")
print(f"Total neurons: {network.count_neurons()}")
print(f"Total links: {network.count_links()}")
print(f"Total multilinks: {network.count_multilinks()}")
```

### Example 2: Layer-by-Layer Analysis

```python
# Analyze connections from each layer
for layer_level in sorted(network.layers.keys()):
    layer = network.layers[layer_level]
    links, multilinks = network.count_connections_from_layer(layer_level)

    layer_type = "Input" if layer_level == 0 else "Output" if layer_level == max(network.layers.keys()) else "Hidden"
    if layer.is_convolutional:
        layer_type += " (Convolutional)"

    print(f"Layer {layer_level} ({layer_type}): {layer.size()} neurons, {links} links, {multilinks} multilinks")
```

### Example 3: Convolutional Layer Map Analysis

```python
# Get detailed map information for convolutional layers
# Note: This also triggers map analysis if not already done
print(network.convolutional_layer_maps_info())

# Alternative: Get full layer info which includes map analysis
print(network.full_layer_info(show_padding=True))

# Access individual layer maps after analysis
for layer_level, layer in network.layers.items():
    if layer.is_convolutional:
        print(f"Layer {layer_level} has {len(layer.maps)} maps:")
        for i, map_obj in enumerate(layer.maps):
            print(f"  Map {i}: {map_obj}")
```

### Example 4: Comprehensive Data Export

```python
# Export all network data to a text file
output_path = network.export_comprehensive_data()
print(f"Comprehensive data exported to: {output_path}")

# Export with custom filename
output_path = network.export_comprehensive_data("detailed_analysis.txt")

# The exported file contains ALL accessible information:
# - Basic network statistics and structure
# - Detailed layer information and properties
# - Complete neuron data and connectivity
# - Link and MultiLink analysis
# - Map information with padding analysis
# - Network topology and connectivity patterns
```

### Example 5: Batch Processing

```python
import glob
from pathlib import Path

# Process multiple XNET files
xnet_files = glob.glob("*.xnet")
for file_path in xnet_files:
    print(f"Processing {file_path}...")
    network = Network(file_path)

    # Export comprehensive data for each network
    output_path = network.export_comprehensive_data()
    print(f"Exported to: {output_path}")
```

## Output

The tool provides two types of output:

### Terminal Output
Text-based analysis displayed in the console including:
- **Network Statistics**: Layer counts, neuron counts, connection counts
- **Layer Information**: Detailed breakdown of each layer's properties
- **Padding Analysis**: Detected padding values for convolutional layers
- **Map Information**: Feature map dimensions and kernel size indicators
- **Connection Analysis**: Link and MultiLink distribution across layers

### File Export
Comprehensive text files containing ALL accessible network data:
- **Complete Coverage**: Every piece of information accessible through the codebase
- **Structured Format**: Clear sections with descriptive headers
- **Detailed Analysis**: Network topology, connectivity patterns, and statistics
- **Permanent Record**: Searchable text format for archival and comparison

## Export File Format

The comprehensive export generates structured text files with the following sections:

1. **Header Information**: Source file, export timestamp
2. **Basic Network Information**: Counts, statistics, layer breakdown
3. **Detailed Layer Information**: Per-layer analysis with types and connections
4. **Neuron Information**: Distribution, connectivity statistics
5. **Link Information**: Regular links vs MultiLinks analysis
6. **MultiLink Block Information**: Block sizes, distribution, sample details
7. **Map Information**: Dimensions, padding, convolutional analysis
8. **Network Connectivity Analysis**: Topology, layer-to-layer connections
9. **Export Summary**: Coverage confirmation

Example output filename: `star4_24_12.xnet` → `star4_24_12.txt`

## Dependencies

This tool uses only Python standard library modules:

- `gzip`: For decompressing XNET files
- `xml.etree.ElementTree`: For XML parsing
- `math`: For mathematical operations
- `datetime`: For export timestamps
- `os`: For file operations
- `sys`: For command line argument handling
- `dataclasses`: For data structure definitions

No external dependencies are required.

## Documentation

- **Main README**: This file - overview and basic usage
- **Code Documentation**: Inline docstrings and comments throughout the codebase
- **Method Documentation**: `methods.txt` - analysis method documentation
- **MultiLink Documentation**: `multilink_blocks.txt` - MultiLink analysis documentation

## Troubleshooting

### Common Issues

1. **"No layer at level X" error**: Check that the network file is valid and contains the expected layer structure
2. **"usage: python main.py <path to xnet>" error**: Provide the path to an XNET file as a command line argument
3. **XML parsing errors**: Ensure the XNET file is not corrupted and is properly gzip-compressed
4. **Import errors**: Ensure you're running from the correct directory and all component files are present
5. **Export permission errors**: Ensure write permissions for the output directory
6. **Large file exports**: For very large networks, export files may be several MB in size
7. **Python command not found**: Use `python3` instead of `python` on systems where Python 3 is not the default

## Performance Notes

- Large networks may take significant time to process during parsing and analysis
- XML decompression creates temporary files that are automatically cleaned up after parsing
- Memory usage scales with network size, particularly for networks with many MultiLinks
- Padding analysis requires additional computation time for convolutional layers
- Export operations are generally fast, but file I/O time scales with data volume

## Advanced Usage

### Custom Analysis

```python
# Analyze specific layers
for layer_level in sorted(network.layers.keys()):
    layer = network.layers[layer_level]
    if layer.level > 0:  # Skip input layer
        links, multilinks = network.count_connections_from_layer(layer.level)
        print(f"Layer {layer.level}: {layer.size()} neurons, {links} links, {multilinks} multilinks")

        # Check if convolutional
        if layer.is_convolutional:
            print(f"  Convolutional layer with {len(layer.maps)} maps")

# Access raw network data
for neuron in network.neurons:
    print(f"Neuron {neuron.id} in layer {neuron.layer}")
    print(f"  Incoming connections: {len(neuron.links_in)}")
    print(f"  Outgoing connections: {len(neuron.links_out)}")

# Access MultiLink information
print(f"Total unique MultiLink IDs: {len(network.unique_multilinks)}")
print(f"Total MultiLink blocks: {len(network.multilink_blocks)}")
```

### Batch Processing

```python
import glob
from pathlib import Path

# Process multiple XNET files
xnet_files = glob.glob("xnet_files/*.xnet")
for file_path in xnet_files:
    print(f"Processing {file_path}...")
    network = Network(file_path)

    # Run analysis
    print(network.generic_info())
    print(network.full_layer_info(show_padding=True))
    print("-" * 80)
```

## Technical Details

### XNET Format Specification

The XNET format uses the following XML structure:
- `<layer>` elements define network layers with class and level attributes
- `<neuron>` elements define individual neurons with weights and positions
- `<link>` elements define connections with source, sink, and weight
- MultiLinks share IDs to represent convolution kernel operations

### Algorithm Details

**Map Analysis**: Feature maps are identified by analyzing patterns in MultiLink block structures and detecting ID jumps that indicate map boundaries. The algorithm assumes square maps and kernels.

**Padding Detection**: Padding is detected by analyzing the size patterns of MultiLink blocks as kernels move across feature maps. The algorithm identifies when kernel visibility changes due to padding constraints.

**MultiLink Organization**: MultiLinks are organized into blocks based on their sink neurons, with each block representing a kernel operation applied to a specific output neuron.

## Contributing

To contribute to this project:

1. Follow the existing code style and documentation patterns
2. Add type hints to all new functions
3. Include docstrings for all public methods
4. Test with the provided sample XNET files
5. Update this README if adding new features

## Known Limitations

- Assumes square feature maps in convolutional layer analysis
- Assumes square kernels for padding detection
- Padding analysis works best when entire kernel is visible in some positions
- Currently optimized for convolutional networks with MultiLink connections

## Future Enhancements

- Support for non-square feature maps and kernels
- Export to standard neural network formats (ONNX, TensorFlow, PyTorch)
- Performance optimizations for very large networks
- Support for additional network architectures beyond CNNs
- Visualization tools for network structure and kernel patterns
- Interactive analysis tools

## Academic Context

This tool was developed for analyzing neural network structures in natural computation research, with particular focus on understanding convolutional neural network architectures through their connection patterns. It has been used to analyze networks trained on image classification tasks and to understand the spatial organization of learned features.

## Citation

If you use this tool in academic work, please cite appropriately according to your institution's guidelines.

## License

This project is intended for academic and research use. Please refer to your institution's policies regarding software usage and distribution rights.
