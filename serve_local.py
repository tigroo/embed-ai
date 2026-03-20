#!/usr/bin/env python3
"""Local HTTPS server for testing the PWA on a phone over the LAN.

Camera access requires a secure context (HTTPS).  This script creates
a temporary self-signed certificate and serves ``docs/`` over HTTPS.

Usage:
    python serve_local.py          # default port 8443
    python serve_local.py 9443     # custom port

Then open on your phone:  https://<your-PC-IP>:8443/pwa/
Accept the browser security warning (self-signed cert) — the camera
will work after that.
"""

import http.server
import mimetypes
import os
import socket
import ssl
import subprocess
import sys
import tempfile


def get_local_ip() -> str:
    """Best-effort LAN IP detection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_self_signed_cert(certfile: str, keyfile: str) -> None:
    """Generate a temporary self-signed cert valid for 1 day."""
    subprocess.check_call([
        "openssl", "req", "-x509",
        "-newkey", "rsa:2048",
        "-keyout", keyfile,
        "-out", certfile,
        "-days", "1",
        "-nodes",
        "-subj", "/CN=embed-ai-local",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class LoggingHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with visible request logging and ONNX MIME."""

    # Register .onnx so the browser doesn't reject it as unknown type
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        # Allow WASM streaming compilation and cross-origin isolation
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")

        # Camera permissions
        self.send_header("Permissions-Policy", "camera=*")

        # Dev server: never cache — avoids stale JS/WASM/model issues
        self.send_header("Cache-Control", "no-store")

        super().end_headers()

    def log_message(self, fmt, *args):
        # Print every request to stdout so we can debug serving issues
        sys.stdout.write(
            f"  [{self.log_date_time_string()}] "
            f"{self.address_string()} - {fmt % args}\n"
        )
        sys.stdout.flush()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    os.chdir(docs_dir)

    # Ensure MIME types are registered globally too
    mimetypes.add_type("application/octet-stream", ".onnx")
    mimetypes.add_type("application/wasm", ".wasm")

    # Generate ephemeral cert
    tmp = tempfile.mkdtemp()
    cert = os.path.join(tmp, "cert.pem")
    key = os.path.join(tmp, "key.pem")
    generate_self_signed_cert(cert, key)

    ip = get_local_ip()

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)

    server = http.server.HTTPServer(("0.0.0.0", port), LoggingHandler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    # List what we will serve
    model_dir = os.path.join("pwa", "models")
    if os.path.isdir(model_dir):
        models = os.listdir(model_dir)
        model_info = ", ".join(
            f"{m} ({os.path.getsize(os.path.join(model_dir, m)) / 1_048_576:.1f} MB)"
            for m in sorted(models)
        )
    else:
        model_info = "(no models/ directory found!)"

    print()
    print("=" * 60)
    print("  PWA local server running")
    print(f"  https://{ip}:{port}/pwa/")
    print()
    print(f"  Models: {model_info}")
    print()
    print("  Open this URL on your phone (same WiFi).")
    print("  Accept the security warning (self-signed cert).")
    print("=" * 60)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

