from dataclasses import dataclass, field

from components.link import MultiLinkBlock
from components.neuron import Neuron


class Map:
    """
    Represents a map of the neural network

    Attributes:
        layer_level: the corresponding layer
        size: the rectangular size of the map
        conn_num: the "cn" attribute (kernel size)
        conv_map: True, if this map is from a convolutional layer
    """

    def __init__(
        self,
        layer: int,
        size: tuple[int, int],
        conn_num: int,
        is_conv_map=False,
    ):
        self.layer_level = layer
        self.size = size
        self.conn_num = conn_num
        self.conv_map = is_conv_map
        self.multilink_blocks: list[MultiLinkBlock] = []
        self.padding: tuple[int, int, int, int] | None = None

    def set_padding(self, top: int, right: int, bottom: int, left: int):
        self.padding = top, right, bottom, left

    def get_block_sizes(self) -> list[int]:
        return list(map(lambda block: len(block), self.multilink_blocks))

    def __repr__(self) -> str:
        lsize, rsize = self.size
        string = f"{lsize}x{rsize} Map, cn={self.conn_num}"
        if self.padding:
            top, right, bottom, left = self.padding
            padding = (
                f", Padding=(Top: {top}, Right: {right}, Bot: {bottom}, Left: {left})"
            )
            return string + padding
        else:
            return string


@dataclass
class Layer:
    """
    Represents a layer of the neural network

    Attributes:
        level: the level of the layer in the network hierarchy
        neurons: the neurons that belong to this layer
        maps: the maps that are associated with this layer
        is_convolutional: True, if this layer is a convolutional layer
    """

    level: int
    neurons: list[Neuron] = field(default_factory=list[Neuron])
    maps: list[Map] = field(default_factory=list[Map])
    is_convolutional: bool = False

    def size(self) -> int:
        return len(self.neurons)
