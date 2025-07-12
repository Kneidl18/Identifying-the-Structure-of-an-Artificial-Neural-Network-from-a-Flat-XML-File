from xnet_parser import Network
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path to xnet> [--export]")
        print("  --export: Export comprehensive data to text file instead of printing")
        exit(1)

    xnet_path = sys.argv[1]
    export_mode = "--export" in sys.argv

    network = Network(xnet_path)

    if export_mode:
        # Export comprehensive data to file
        base_name = os.path.splitext(os.path.basename(xnet_path))[0]
        output_path = f"{base_name}.txt"

        result_path = network.export_comprehensive_data(output_path)
        print(f"Comprehensive network data exported to: {result_path}")

        # Show file size
        file_size = os.path.getsize(result_path)
        print(f"Export file size: {file_size:,} bytes")
    else:
        # Original behavior - print to terminal
        print(network.full_layer_info(show_padding=True))


if __name__ == "__main__":
    main()
