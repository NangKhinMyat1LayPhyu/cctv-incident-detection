"""One command for all tools: incident-recon <command> [options].

Each command runs the matching script in scripts/ with the same options (see
`incident-recon <command> --help`), so the scripts also keep working on their own.
Needs the repository checkout: install with `pip install -e .`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

COMMANDS = {
    "analyze": ("analyze.py", "one clip: tracking + incident report"),
    "live": ("live.py", "live incident detection on one or more streams, with alerts"),
    "serve": ("serve.py", "web service and dashboard"),
    "track": ("run_baseline.py", "tracking only, for a folder of clips"),
    "eval": ("eval_model.py", "run a detector on the held-out clips"),
    "score": ("eval_heldout.py", "recall / precision / false alarms per hour from eval results"),
    "export": ("export.py", "export the detector to ONNX and CoreML"),
    "bench": ("bench_models.py", "speed and box agreement of exported detectors"),
    "demo": ("make_demo.py", "render a demo GIF of an analysed clip"),
    "train": ("train.py", "fine-tune the detector"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:{width}}  {about}" for name, (_, about) in COMMANDS.items()]
    return "usage: incident-recon <command> [options]\n\ncommands:\n" + "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage())
        return
    name = sys.argv[1]
    if name not in COMMANDS:
        sys.exit(f"unknown command: {name}\n\n{usage()}")
    script = SCRIPTS / COMMANDS[name][0]
    if not script.exists():
        sys.exit(f"{script} not found: incident-recon needs the repository checkout (pip install -e .)")
    # Run the script as __main__. Not runpy.run_path: it would replace sys.argv[0] with the
    # script's path, and --help would then say "live.py" instead of "incident-recon live".
    sys.argv = [f"incident-recon {name}", *sys.argv[2:]]
    code = compile(script.read_text(), str(script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script)})


if __name__ == "__main__":
    main()
