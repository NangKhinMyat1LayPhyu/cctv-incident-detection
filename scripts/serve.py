"""Start the web service and dashboard (see incident_recon/service.py).

Run the live process with the same --out folder as --live-dir here, then open the dashboard.

Usage:
  python scripts/serve.py --live-dir outputs/live --weights best.pt
  open http://127.0.0.1:8000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from incident_recon.service import create_app  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--live-dir", default="outputs/live", help="The live process's --out folder")
    p.add_argument("--jobs-dir", default="outputs/jobs", help="Where uploaded clips and their results go")
    p.add_argument("--weights", default=None, help="Detector for uploaded clips (without it, uploads are refused)")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only (no login on the service)")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    weights = str(Path(args.weights).expanduser()) if args.weights else None
    app = create_app(args.live_dir, args.jobs_dir, weights=weights)
    print(f"Dashboard: http://{args.host}:{args.port}  (live folder: {args.live_dir})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
