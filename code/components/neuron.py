from dataclasses import dataclass, field
from components.link import Link


@dataclass
class Neuron:
    """
    Represents a single neuron in the neural network

    Attributes:
        id: Unique identifier for the neuron
        layer: Layer identifier (as string)
        input: Whether this is an input neuron (metadata)
        links_in: dictionary of incoming connections (source_id -> Link)
        links_out: dictionary of outgoing connections (sink_id -> Link)

    We tried to identify a neuron's layer, but could not find a reliable way to do so
    """

    id: int
    layer: int
    links_in: dict[int, Link] = field(default_factory=dict)
    links_out: dict[int, Link] = field(default_factory=dict)
