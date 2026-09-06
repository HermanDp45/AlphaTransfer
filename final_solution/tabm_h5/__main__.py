import sys

from final_solution.core import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "h5", *sys.argv[1:]]))
