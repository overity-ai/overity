"""
Simple HTML server to display reports
=====================================

**May 2025**

- Florian Dupeyron (florian.dupeyron@elsys-design.com)

> This file is part of the Overity.ai project, and is licensed under
> the terms of the Apache 2.0 license. See the LICENSE file for more
> information.
"""

import logging

log = logging.getLogger("backend.report_view.simple_server")


def serve(html_string: str, port: int = 8000):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_string.encode())

    httpd = HTTPServer(("localhost", port), Handler)
    log.info(f"Serving at http://localhost:{port}... Press [Ctrl+C] to stop")
    httpd.serve_forever()
