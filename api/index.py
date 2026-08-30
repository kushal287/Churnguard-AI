"""
Vercel Serverless Entrypoint for ChurnGuard AI.
Handles incoming HTTP requests and serves the application gateway.
"""

from http.server import BaseHTTPRequestHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"


class handler(BaseHTTPRequestHandler):
    """Vercel standard Python serverless request handler."""

    def do_GET(self):
        path_clean = self.path.split("?")[0].strip("/")
        
        # Route static assets
        if path_clean in ["favicon.ico", "favicon.png", "logo.png"]:
            file_path = PUBLIC_DIR / path_clean
            if not file_path.exists():
                file_path = BASE_DIR / "app" / "assets" / (path_clean if path_clean != "logo.png" else "logo.png")
            
            if file_path.exists():
                self.send_response(200)
                content_type = "image/x-icon" if path_clean.endswith(".ico") else "image/png"
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Serve main index.html
        index_file = PUBLIC_DIR / "index.html"
        if not index_file.exists():
            index_file = BASE_DIR / "index.html"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        
        if index_file.exists():
            with open(index_file, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.wfile.write(b"<!DOCTYPE html><html><body><h1>ChurnGuard AI is Live</h1></body></html>")
