"""Run the installed ShardScope benchmark interface."""

import sys

from shardscope.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare", *sys.argv[1:]]))
