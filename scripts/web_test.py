"""Check a built browser player (mise web:build) in headless Chrome.

Usage: uv run --locked python scripts/web_test.py web/dist

Serves the build, opens index.html?test in headless Chrome (CHROME, or the first Chrome or
Chromium on PATH) and waits for the page's report: the GPU network's largest gaps from PyTorch
on the exported reference positions, then a full c4a0-vs-c4a0 game at a small budget. Without
a GPU, Chrome runs WebGPU on SwiftShader, which is slow but exact enough for the check.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOLERANCE = 1e-3  # float32 GPU sums against PyTorch on the CPU
QUERY = (
    "test&iterations=32&depth=4"  # a quick full game; the page defaults are 256 and 10
)
CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def find_chrome() -> str:
    for name in (os.environ.get("CHROME"), *CHROMES):
        if name and (path := shutil.which(name)):
            return path
    sys.exit(f"no Chrome found: set CHROME or install one of {', '.join(CHROMES)}")


def main() -> None:
    site = Path(sys.argv[1]).resolve()
    reports: list[dict[str, object]] = []
    done = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(204)
            self.end_headers()
            if self.path == "/result":  # the page also posts per-move /progress
                reports.append(json.loads(body))
                done.set()

    class Server(ThreadingHTTPServer):
        def handle_error(self, request: object, client_address: object) -> None:
            if not isinstance(
                sys.exc_info()[1], ConnectionResetError
            ):  # Chrome exiting
                super().handle_error(request, client_address)  # type: ignore[arg-type]

    server = Server(("127.0.0.1", 0), partial(Handler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html?{QUERY}"
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [
                find_chrome(),
                "--headless=new",
                "--no-sandbox",
                "--enable-unsafe-webgpu",
                f"--user-data-dir={profile}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            finished = done.wait(timeout=600)
        finally:
            chrome.kill()
            chrome.wait()
            server.shutdown()
    if not finished:
        sys.exit("the page never reported")
    report = reports[0]
    print(json.dumps(report))
    if "error" in report:
        sys.exit(f"page error: {report['error']}")
    check = report["check"]
    assert isinstance(check, dict)
    assert check["policy"] < TOLERANCE, (
        f"policy differs from PyTorch by {check['policy']}"
    )
    assert check["value"] < TOLERANCE, f"value differs from PyTorch by {check['value']}"
    assert report["outcome"] in ("win", "draw"), f"game did not finish: {report}"
    print(f"ok: {report['adapter']}, {len(str(report['moves']))}-move game")


if __name__ == "__main__":
    main()
