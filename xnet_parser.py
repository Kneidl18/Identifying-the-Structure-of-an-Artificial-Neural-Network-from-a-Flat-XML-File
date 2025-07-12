import gzip
import math
import xml.etree.ElementTree as ET
import os

from components.neuron import Neuron
from components.layer import Layer, Map
from components.link import Link, MultiLinkBlock
import util


class Network:
    """
    Main class for parsing and analyzing neural networks from XNET files.

    XNET files contain compressed XML representations of neural networks.
    This class extracts the network structure including layers, neurons,
    and connections, with special focus on convolutional layer analysis.
    """

    def __init__(self, xnet_path: str):
        """
        Instantiates and parses a new network from the given file

        Args:
            xnet_path: the path to the xnet file that is to be parsed
        """
        self.xnet_path = xnet_path

        # layers (requires the "ly" attribute)
        self.layers: dict[int, Layer] = {}
        self.maps_analyzed = False  # analyze_maps

        # neurons
        self.neurons: list[Neuron] = []  # to keep original order
        self.neuron_index: dict[int, Neuron] = {}  # for fast access

        # links
        self.links: list[Link] = []  # contains non-multilinks
        self.multilinks: list[Link] = []
        self.unique_multilinks: dict[int, list] = {}
        self.multilink_blocks: list[MultiLinkBlock] = []

        # Parse the network from the file
        self._parse_xnet()

    """
    --------------------------
    Parser
    --------------------------
    """

    def _parse_xnet(self):
        """
        Parses the xnet file to gather basic information for further analysis

        Raises:
            RuntimeError: if the parsing fails
        """
        # Decompress the XNET file
        decompressed_path = self.xnet_path + ".xml"
        try:
            with gzip.open(self.xnet_path, "rb") as f_in:
                with open(decompressed_path, "wb") as f_out:
                    f_out.write(f_in.read())

            # Parse the XML
            tree = ET.parse(decompressed_path)
            root = tree.getroot()

            # Extract network components
            self._extract_io_layers(root)
            self._extract_neurons(root)
            self._extract_links(root)
        except Exception as e:
            raise RuntimeError(f"Failed to parse XNET file {self.xnet_path}: {e}")
        finally:
            # Clean up temporary XML file
            if os.path.exists(decompressed_path):
                os.remove(decompressed_path)

    def _extract_io_layers(self, root: ET.Element):
        """
        Extracts information on the Input and Output layers,
        since they are to be used as input parameters

        Args:
            root: the root of the XML parse tree
        """
        layers_metadata = root.findall(".//layer")
        first_last_elems = [layers_metadata[0], layers_metadata[-1]]

        for elem in first_last_elems:
            layer = Layer(int(elem.get("level", 0)))

            # map size
            map_ = elem.findall("map")[0]
            size = map_.get("size", "").split("x")
            map_size = (int(size[0]), int(size[1]))

            # size of kernel
            cn = int(map_.get("cn", 0))

            layer_map = Map(
                layer=layer.level,
                size=map_size,
                conn_num=cn,
            )
            layer.maps.append(layer_map)

            # add to layers
            self.layers[layer.level] = layer

    def _extract_neurons(self, root: ET.Element):
        """
        Extracts all neurons from the network and adds them
        to their corresponding layer (using the "ly" attribute)

        Args:
            root: the root of the XML parse tree
        """
        for neuron_elem in root.findall(".//neuron"):
            neuron_attrs = dict(neuron_elem.attrib)

            # new neuron
            neuron = Neuron(
                id=int(neuron_attrs.get("id", 0)),
                # we require the "ly" attribute for layer analysis
                layer=int(neuron_attrs.get("ly", "0")),
            )

            # add neuron to collection
            self.neurons.append(neuron)
            self.neuron_index[neuron.id] = neuron

            # add neuron to its layer (requires the "ly" attribute)
            if neuron.layer not in self.layers:
                self.layers[neuron.layer] = Layer(neuron.layer)
            self.layers[neuron.layer].neurons.append(neuron)

    def _extract_links(self, root: ET.Element) -> None:
        """
        Extracts all links from the network. Also organises the MultiLinks
        for further analysis

        Args:
            root: the root of the XML parse tree
        """
        for link_elem in root.findall(".//link"):
            class_name = link_elem.get("class", "")

            # Create link object
            link = Link(
                class_name=class_name,
                id=int(link_elem.get("id", 0)),
                source_id=int(link_elem.get("so", 0)),
                sink_id=int(link_elem.get("si", 0)),
            )

            # Add to links collection
            if class_name == "MultiLink":
                self.multilinks.append(link)

                if link.id not in self.unique_multilinks:
                    self.unique_multilinks[link.id] = []
                self.unique_multilinks[link.id].append(link)
            else:
                self.links.append(link)

            # Add link references to neurons
            source_neuron = self.neuron_index.get(link.source_id)
            sink_neuron = self.neuron_index.get(link.sink_id)

            if source_neuron:
                source_neuron.links_out[link.sink_id] = link
            if sink_neuron:
                sink_neuron.links_in[link.source_id] = link

                # if Multilink, then Layer is convolutional layer
                if class_name == "MultiLink":
                    self.layers[sink_neuron.layer].is_convolutional = True

        # Organize MultiLinks into blocks for analysis
        self._organize_multilink_blocks()

    def _organize_multilink_blocks(self) -> None:
        """
        Organizes MultiLinks into blocks for analysis of convolutional layers

        Groups consecutive MultiLinks that share the same sink neuron into
        MultiLinkBlocks, which each represent a kernel operation of a map in convolutional layers
        """
        current_sink = None

        for link in self.multilinks:
            if link.sink_id != current_sink:
                # Start a new block
                current_sink = link.sink_id
                self.multilink_blocks.append(MultiLinkBlock(current_sink))

            # Add to current block
            self.multilink_blocks[-1].append_multilink(link)

    """
    -------------------------------------
    Convolutional Layer Analysis
    -------------------------------------
    """

    def _analyze_conv_layer_maps(self):
        """
        Analyzes the network's maps of its convolutional layers and puts them into their respective layers.

        Assumptions:
            - each map uses an entirely different set of multilinks for their kernel
            - maps are square
            - kernels are square
        """
        # init with first block
        mlink_pool = set(self.multilink_blocks[0].get_all_ids())
        current_blocks = []
        current_layer = 0

        for block in self.multilink_blocks:
            # prev ∩ current == ∅  => new map starts
            if not mlink_pool.intersection(block.get_all_ids()):
                # construct new map
                mlink_count = len(mlink_pool)
                map_len = int(math.sqrt(len(current_blocks)))
                map_ = Map(
                    layer=current_layer,
                    size=(map_len, map_len),
                    conn_num=int(math.sqrt(mlink_count)),
                    is_conv_map=True,
                )
                map_.multilink_blocks.extend(current_blocks)
                # add to corresponding layer
                self.layers[current_layer].maps.append(map_)

                # reset vars for next map
                mlink_pool.clear()
                mlink_pool = mlink_pool.union(block.get_all_ids())
                current_blocks = []

            mlink_pool = mlink_pool.union(block.get_all_ids())
            current_blocks.append(block)
            current_layer = self.neuron_index[block.sink].layer

        # end of last map, not covered by for loop
        if mlink_pool:
            # construct new map
            mlink_count = len(mlink_pool)
            map_len = int(math.sqrt(len(current_blocks)))
            map_ = Map(
                layer=current_layer,
                size=(map_len, map_len),
                conn_num=int(math.sqrt(mlink_count)),
                is_conv_map=True,
            )
            # add to corresponding layer
            self.layers[current_layer].maps.append(map_)

        self.maps_analyzed = True

    def _analyze_conv_layer_padding(self):
        """Analyzes the padding of maps of convolutional layers"""

        # we need the maps to be analysed before trying to determine padding
        if not self.maps_analyzed:
            self._analyze_conv_layer_maps()

        for level in sorted(self.layers.keys()):
            layer = self.layers[level]

            # we only consider convolutional layers
            if not layer.is_convolutional:
                continue

            for map_ in layer.maps:
                # algorithm only requires ordered sizes of MultiLinkBlocks
                block_sizes = map_.get_block_sizes()

                # count from top-left/bottom-right
                left_padding = util.count_increase(block_sizes)
                right_padding = util.count_increase(list(reversed(block_sizes)))

                # only consider blocks unaffected by left/right padding
                repeated_sizes = util.keep_repeats_and_group(block_sizes)

                top_padding = util.count_increase(repeated_sizes)
                bot_padding = util.count_increase(list(reversed(repeated_sizes)))

                map_.set_padding(
                    top=top_padding,
                    right=right_padding,
                    bottom=bot_padding,
                    left=left_padding,
                )

    """
    -----------------------------------
    General Information
    -----------------------------------
    """

    def count_layers(self):
        return len(self.layers)

    def count_neurons(self) -> int:
        return len(self.neuron_index)

    def count_anylink(self):
        return self.count_links() + self.count_multilinks()

    def count_links(self):
        return len(self.links)

    def count_multilinks(self):
        return len(self.multilinks)

    def count_unique_multilinks(self):
        return len(self.unique_multilinks)

    def count_connections_from_layer(self, layer_level: int) -> tuple[int, int]:
        """
        Counts the number of links and multilinks from a given layer

        Args:
            layer_level: the specified layer

        Returns: #Links and #MultiLinks from that layer as a tuple
        """
        if layer_level not in self.layers:
            return 0, 0

        layer = self.layers[layer_level]
        count_links = 0
        count_multilinks = 0

        for neuron in layer.neurons:
            for link in neuron.links_out.values():
                if link.class_name == "MultiLink":
                    count_multilinks += 1
                else:
                    count_links += 1

        return count_links, count_multilinks

    """
    -----------------------------------
    Methods for pre-formatted output
    -----------------------------------
    """

    def export_comprehensive_data(self, output_path: str | None = None) -> str:
        """
        Exports all available neural network data to a comprehensive text file.

        Args:
            output_path: Optional custom output path. If None, generates from input filename.

        Returns:
            The path to the generated output file.
        """
        import os

        # Generate output filename if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.xnet_path))[0]
            output_path = f"{base_name}.txt"

        # Ensure maps are analyzed for complete data
        if not self.maps_analyzed:
            self._analyze_conv_layer_maps()
        self._analyze_conv_layer_padding()

        # Collect all information
        content_sections = []

        # Header section
        content_sections.append("=" * 80)
        content_sections.append("COMPREHENSIVE NEURAL NETWORK DATA EXPORT")
        content_sections.append("=" * 80)
        content_sections.append(f"Source File: {self.xnet_path}")
        content_sections.append(f"Export Date: {self._get_current_timestamp()}")
        content_sections.append("")

        # Basic network information
        content_sections.append(self._export_basic_network_info())

        # Detailed layer information
        content_sections.append(self._export_detailed_layer_info())

        # Neuron information
        content_sections.append(self._export_neuron_info())

        # Link information
        content_sections.append(self._export_link_info())

        # MultiLink analysis
        content_sections.append(self._export_multilink_info())

        # Map information
        content_sections.append(self._export_map_info())

        # Network connectivity analysis
        content_sections.append(self._export_connectivity_analysis())

        # Write to file
        full_content = "\n".join(content_sections)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return output_path

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for export header."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _export_basic_network_info(self) -> str:
        """Export basic network statistics and information."""
        lines = []
        lines.append("BASIC NETWORK INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Total Layers: {self.count_layers()}")
        lines.append(f"Total Neurons: {self.count_neurons():,}")
        lines.append(f"Total Links: {self.count_links():,}")
        lines.append(f"Total MultiLinks: {self.count_multilinks():,}")
        lines.append(f"Unique MultiLinks: {self.count_unique_multilinks():,}")
        lines.append(f"Total Connections: {self.count_anylink():,}")
        lines.append(f"Maps Analyzed: {self.maps_analyzed}")
        lines.append("")

        # Layer type breakdown
        conv_layers = sum(1 for layer in self.layers.values() if layer.is_convolutional)
        input_output_layers = 2  # First and last
        other_layers = len(self.layers) - conv_layers - input_output_layers

        lines.append("Layer Type Breakdown:")
        lines.append(f"  Input/Output Layers: {input_output_layers}")
        lines.append(f"  Convolutional Layers: {conv_layers}")
        lines.append(f"  Other Layers: {other_layers}")
        lines.append("")

        return "\n".join(lines)

    def _export_detailed_layer_info(self) -> str:
        """Export detailed information about each layer."""
        lines = []
        lines.append("DETAILED LAYER INFORMATION")
        lines.append("-" * 40)

        first_level = min(self.layers.keys()) if self.layers else 0
        last_level = max(self.layers.keys()) if self.layers else 0

        for level in sorted(self.layers.keys()):
            layer = self.layers[level]
            lines.append(f"Layer {level}:")
            lines.append(f"  Neuron Count: {layer.size()}")

            # Layer type
            if level == first_level:
                layer_type = "Input Layer"
            elif level == last_level:
                layer_type = "Output Layer"
            elif layer.is_convolutional:
                layer_type = "Convolutional Layer"
            else:
                layer_type = "Hidden Layer"
            lines.append(f"  Type: {layer_type}")

            # Connection information
            link_count, mlink_count = self.count_connections_from_layer(level)
            lines.append(f"  Outgoing Links: {link_count}")
            lines.append(f"  Outgoing MultiLinks: {mlink_count}")

            # Map information
            lines.append(f"  Associated Maps: {len(layer.maps)}")
            for i, map_obj in enumerate(layer.maps):
                lines.append(f"    Map {i}: {map_obj.size[0]}x{map_obj.size[1]}, cn={map_obj.conn_num}")
                if map_obj.padding:
                    top, right, bottom, left = map_obj.padding
                    lines.append(f"      Padding: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

            lines.append("")

        return "\n".join(lines)

    def _export_neuron_info(self) -> str:
        """Export detailed neuron information."""
        lines = []
        lines.append("NEURON INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Total Neurons: {len(self.neurons)}")
        lines.append("")

        # Neuron distribution by layer
        neuron_by_layer = {}
        for neuron in self.neurons:
            if neuron.layer not in neuron_by_layer:
                neuron_by_layer[neuron.layer] = []
            neuron_by_layer[neuron.layer].append(neuron)

        lines.append("Neurons by Layer:")
        for layer_id in sorted(neuron_by_layer.keys()):
            neurons_in_layer = neuron_by_layer[layer_id]
            lines.append(f"  Layer {layer_id}: {len(neurons_in_layer)} neurons")

            # Sample neuron IDs (first 10 and last 10 if more than 20)
            neuron_ids = [n.id for n in neurons_in_layer]
            if len(neuron_ids) <= 20:
                lines.append(f"    IDs: {neuron_ids}")
            else:
                lines.append(f"    IDs: {neuron_ids[:10]} ... {neuron_ids[-10:]} (showing first/last 10)")

        lines.append("")

        # Connection statistics
        lines.append("Neuron Connection Statistics:")
        total_incoming = sum(len(neuron.links_in) for neuron in self.neurons)
        total_outgoing = sum(len(neuron.links_out) for neuron in self.neurons)
        lines.append(f"  Total Incoming Connections: {total_incoming}")
        lines.append(f"  Total Outgoing Connections: {total_outgoing}")

        if self.neurons:
            avg_incoming = total_incoming / len(self.neurons)
            avg_outgoing = total_outgoing / len(self.neurons)
            lines.append(f"  Average Incoming per Neuron: {avg_incoming:.2f}")
            lines.append(f"  Average Outgoing per Neuron: {avg_outgoing:.2f}")

        lines.append("")
        return "\n".join(lines)

    def _export_link_info(self) -> str:
        """Export detailed link information."""
        lines = []
        lines.append("LINK INFORMATION")
        lines.append("-" * 40)

        # Basic link statistics
        lines.append(f"Regular Links: {len(self.links)}")
        lines.append(f"MultiLinks: {len(self.multilinks)}")
        lines.append(f"Unique MultiLink IDs: {len(self.unique_multilinks)}")
        lines.append("")

        # Regular links analysis
        if self.links:
            lines.append("Regular Links Analysis:")
            link_classes = {}
            for link in self.links:
                if link.class_name not in link_classes:
                    link_classes[link.class_name] = 0
                link_classes[link.class_name] += 1

            for class_name, count in link_classes.items():
                lines.append(f"  {class_name}: {count}")
            lines.append("")

        # MultiLinks analysis
        if self.multilinks:
            lines.append("MultiLinks Analysis:")
            lines.append(f"  Total MultiLink instances: {len(self.multilinks)}")
            lines.append(f"  Unique MultiLink groups: {len(self.unique_multilinks)}")

            # MultiLink group sizes
            group_sizes = [len(group) for group in self.unique_multilinks.values()]
            if group_sizes:
                lines.append(f"  Average group size: {sum(group_sizes) / len(group_sizes):.2f}")
                lines.append(f"  Min group size: {min(group_sizes)}")
                lines.append(f"  Max group size: {max(group_sizes)}")
            lines.append("")

        return "\n".join(lines)

    def _export_multilink_info(self) -> str:
        """Export detailed MultiLink block information."""
        lines = []
        lines.append("MULTILINK BLOCK INFORMATION")
        lines.append("-" * 40)

        if not self.multilink_blocks:
            lines.append("No MultiLink blocks found.")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"Total MultiLink Blocks: {len(self.multilink_blocks)}")

        # Block size statistics
        block_sizes = [len(block) for block in self.multilink_blocks]
        if block_sizes:
            lines.append(f"Average block size: {sum(block_sizes) / len(block_sizes):.2f}")
            lines.append(f"Min block size: {min(block_sizes)}")
            lines.append(f"Max block size: {max(block_sizes)}")

            # Block size distribution
            size_distribution = {}
            for size in block_sizes:
                size_distribution[size] = size_distribution.get(size, 0) + 1

            lines.append("Block size distribution:")
            for size in sorted(size_distribution.keys()):
                count = size_distribution[size]
                lines.append(f"  Size {size}: {count} blocks")

        lines.append("")

        # Sample block details (first 10 blocks)
        lines.append("Sample Block Details (first 10):")
        for i, block in enumerate(self.multilink_blocks[:10]):
            lines.append(f"  Block {i}: Size={len(block)}, Sink={block.sink}")
            # Show first few MultiLink IDs in the block
            link_ids = [link.id for link in block.multi_links[:5]]
            if len(block.multi_links) > 5:
                lines.append(f"    Link IDs: {link_ids}... (showing first 5)")
            else:
                lines.append(f"    Link IDs: {link_ids}")

        lines.append("")
        return "\n".join(lines)

    def _export_map_info(self) -> str:
        """Export detailed map information."""
        lines = []
        lines.append("MAP INFORMATION")
        lines.append("-" * 40)

        total_maps = sum(len(layer.maps) for layer in self.layers.values())
        lines.append(f"Total Maps: {total_maps}")
        lines.append("")

        if total_maps == 0:
            lines.append("No maps found.")
            lines.append("")
            return "\n".join(lines)

        # Maps by layer
        for level in sorted(self.layers.keys()):
            layer = self.layers[level]
            if layer.maps:
                lines.append(f"Layer {level} Maps:")
                for i, map_obj in enumerate(layer.maps):
                    lines.append(f"  Map {i}:")
                    lines.append(f"    Size: {map_obj.size[0]}x{map_obj.size[1]}")
                    lines.append(f"    Connection Number (cn): {map_obj.conn_num}")
                    lines.append(f"    Is Convolutional: {map_obj.conv_map}")
                    lines.append(f"    Associated MultiLink Blocks: {len(map_obj.multilink_blocks)}")

                    if map_obj.padding:
                        top, right, bottom, left = map_obj.padding
                        lines.append(f"    Padding: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
                    else:
                        lines.append("    Padding: Not analyzed")

                    # MultiLink block details for this map
                    if map_obj.multilink_blocks:
                        block_sizes = [len(block) for block in map_obj.multilink_blocks]
                        lines.append(f"    Block sizes: {block_sizes[:10]}{'...' if len(block_sizes) > 10 else ''}")

                lines.append("")

        return "\n".join(lines)

    def _export_connectivity_analysis(self) -> str:
        """Export network connectivity analysis."""
        lines = []
        lines.append("NETWORK CONNECTIVITY ANALYSIS")
        lines.append("-" * 40)

        # Layer-to-layer connectivity
        lines.append("Layer-to-Layer Connectivity:")
        for level in sorted(self.layers.keys()):
            link_count, mlink_count = self.count_connections_from_layer(level)
            total_connections = link_count + mlink_count

            if total_connections > 0:
                lines.append(f"  Layer {level} -> Next Layer:")
                lines.append(f"    Regular Links: {link_count}")
                lines.append(f"    MultiLinks: {mlink_count}")
                lines.append(f"    Total: {total_connections}")

        lines.append("")

        # Network topology summary
        lines.append("Network Topology Summary:")
        if self.layers:
            input_layer = min(self.layers.keys())
            output_layer = max(self.layers.keys())
            hidden_layers = len(self.layers) - 2

            lines.append(f"  Input Layer: {input_layer} ({self.layers[input_layer].size()} neurons)")
            lines.append(f"  Hidden Layers: {hidden_layers}")
            lines.append(f"  Output Layer: {output_layer} ({self.layers[output_layer].size()} neurons)")
            lines.append(f"  Network Depth: {len(self.layers)} layers")

        lines.append("")

        # Convolutional analysis summary
        conv_layers = [level for level, layer in self.layers.items() if layer.is_convolutional]
        if conv_layers:
            lines.append("Convolutional Layer Analysis:")
            lines.append(f"  Convolutional Layers: {conv_layers}")

            total_conv_maps = sum(len(self.layers[level].maps) for level in conv_layers)
            lines.append(f"  Total Convolutional Maps: {total_conv_maps}")

            # Kernel size analysis
            kernel_sizes = []
            for level in conv_layers:
                for map_obj in self.layers[level].maps:
                    if map_obj.conv_map:
                        kernel_sizes.append(map_obj.conn_num)

            if kernel_sizes:
                unique_kernel_sizes = list(set(kernel_sizes))
                lines.append(f"  Kernel Sizes Used: {sorted(unique_kernel_sizes)}")

        lines.append("")

        # Final summary
        lines.append("EXPORT SUMMARY")
        lines.append("-" * 40)
        lines.append("This export contains all available information extracted from the neural network:")
        lines.append("- Basic network statistics and structure")
        lines.append("- Detailed layer information and properties")
        lines.append("- Complete neuron data and connectivity")
        lines.append("- Link and MultiLink analysis")
        lines.append("- Map information with padding analysis")
        lines.append("- Network topology and connectivity patterns")
        lines.append("")
        lines.append("All numerical values, metadata, and structural information")
        lines.append("accessible through the current codebase have been included.")
        lines.append("")

        return "\n".join(lines)

    def generic_info(self) -> str:
        title = f"Network( {self.xnet_path}:"
        info = (
            f"#Layers= {self.count_layers()}, "
            f"#Neurons= {self.count_neurons():,.0f}, "
            f"#Links= {self.count_links():,.0f}, "
            f"#MultiLinks= {self.count_multilinks()} "
            f"({self.count_unique_multilinks():,.0f} unique links) )"
        )
        return f"{'-' * len(info)} \n{title}\n{info} \n{'-' * len(info)}\n"

    def convolutional_layer_maps_info(self) -> str:
        strings: list[str] = []
        for level in sorted(self.layers.keys()):
            strings.append(f"Level {level}:\n")
            for map in self.layers[level].maps:
                strings.append(map.__repr__())
                strings.append("\n")
        return "".join(strings)

    def full_layer_info(self, show_padding=True) -> str:
        if not self.maps_analyzed:
            self._analyze_conv_layer_maps()

        if show_padding:
            self._analyze_conv_layer_padding()

        strings: list[str] = []
        strings.append(self.generic_info())

        first_level = 0
        last_level = len(self.layers) - 1

        for level in sorted(self.layers.keys()):
            layer_size = self.layers[level].size()
            strings.append(f"Layer {level}: {layer_size} Neurons")

            if level == first_level:
                strings.append(" (Input Layer)")
            elif level == last_level:
                strings.append(" (Output Layer)")
            elif self.layers[level].is_convolutional:
                strings.append(" (Convolutional Layer)")
            strings.append("\n")

            link_count, mlink_count = self.count_connections_from_layer(level)
            if level < last_level:
                strings.append("  |\n")
                strings.append("  | ")
                strings.append(
                    f"{link_count} Links"
                    if link_count != 0
                    else f"{mlink_count} MultiLinks"
                )
                strings.append("\n  v\n")

        strings.append("\n")
        strings.append(self.convolutional_layer_maps_info())

        return "".join(strings)
