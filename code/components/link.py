from dataclasses import dataclass


@dataclass
class Link:
    """
    Represents a connection between two neurons

    Attributes:
        class_name: Type of link ("Link" or "MultiLink")
        id: Unique identifier for the link
        source: Source neuron ID
        sink: Destination neuron ID
    """

    class_name: str
    id: int
    source_id: int
    sink_id: int


class MultiLinkBlock:
    """
    A MultiLink Block represents a continuous collection of MultiLinks that
    share the same sink neuron.

    In convolutional neural networks, a single MultiLinkBlock represents a kernel
    operation applied to a specific output neuron. Therefore, this datastructure
    is essential for determining maps, map sizes and kernel sizes.

    Attributes:
        multi_links: Ordered list of MultiLinks in this block
        sink: The destination neuron ID that all links in this block connect to
    """

    def __init__(self, sink: int) -> None:
        """
        Initialize a new MultiLinkBlock.

        Args:
            sink: The destination neuron ID for this block
        """
        self.multi_links: list[Link] = []
        self.sink = sink

    def append_multilink(self, mlink: Link) -> None:
        """
        Adds a MultiLink to this block. This MultiLink must have
        the same sink as specified by this particular MultiLinkBlock.

        Args:
            mlink: the MultiLink that is added to this block
        """
        assert mlink.class_name == "MultiLink", (
            "Only MultiLinks can be added to MultiLinkBlock"
        )
        assert mlink.sink_id == self.sink, (
            f"Link sink {mlink.sink_id} doesn't match block sink {self.sink}"
        )
        self.multi_links.append(mlink)

    def get_all_ids(self) -> set[int]:
        return set(map(lambda link: link.id, self.multi_links))

    def __len__(self) -> int:
        """Return the number of MultiLinks in this block."""
        return len(self.multi_links)

    def __repr__(self) -> str:
        """Return string representation of this block."""
        return f"MultiLinkBlock(size={len(self.multi_links)}, sink={self.sink})"
