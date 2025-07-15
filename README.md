# Neural Network Analysis Tool

A Python tool for analyzing neural networks from XNET files (compressed XML), with focus on convolutional neural network analysis and padding detection.

## Features

- Parse XNET files and extract network structure
- Analyze convolutional layers and detect padding parameters
- Export comprehensive network data to text files
- Analyze MultiLink connections for convolution operations

## Requirements

- Python 3.11 or higher
- No external dependencies (uses Python standard library only)

## Quick Start

```bash
cd code

# Analyze network and print to terminal
python3 main.py path/to/network.xnet

# Export analysis to text file
python3 main.py path/to/network.xnet --export
```

## Python API

```python
from xnet_parser import Network

# Load and analyze network
network = Network("path/to/network.xnet")
print(network.generic_info())
print(network.full_layer_info(show_padding=True))

# Export comprehensive data
output_path = network.export_comprehensive_data()
```

## File Structure

```
code/
├── main.py              # Main analysis script
├── export_network_data.py # Data export script
├── xnet_parser.py       # XNET parser and Network class
├── components/          # Core data structures
└── xnet_files/          # Sample XNET files
```

## Sample Files

The `xnet_files/` directory contains sample networks for testing:
- `star3_208_54.xnet`
- `star4_24_12.xnet`
- `star7_109_15.xnet`

## XNET Format

XNET files are gzip-compressed XML files containing:
- **Layers**: Network layer definitions
- **Neurons**: Individual neuron specifications
- **Links**: Regular connections between neurons
- **MultiLinks**: Shared connections for convolution operations
- **Maps**: Feature map definitions for convolutional layers
