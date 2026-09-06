import argparse

from .mcp.server import AgentServer


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspace",
        required=True,
        help="Path to the project workspace",
    )

    args = parser.parse_args()

    server = AgentServer(args.workspace)
    server.run()


if __name__ == "__main__":
    main()
