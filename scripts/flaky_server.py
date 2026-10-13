"""A tiny local site that answers 429 the first time it sees each page.

    uv run python scripts/flaky_server.py            # serves http://127.0.0.1:8777/
    findex crawl http://127.0.0.1:8777/ --max-pages 5 --log crawl_429.log

Used to show the retry path with real log lines, without bothering a real site.
"""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PAGES = {
    "/": "<html><head><title>Головна</title></head><body>"
    "<a href='/a'>a</a> <a href='/b'>b</a></body></html>",
    "/a": "<html><head><title>A</title></head><body>сторінка a</body></html>",
    "/b": "<html><head><title>B</title></head><body>сторінка b</body></html>",
}
seen: set[str] = set()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # the name is fixed by BaseHTTPRequestHandler
        path = self.path.split("#")[0].rstrip("/") or "/"
        if path == "/robots.txt":
            self.send_response(404)
            self.end_headers()
            return
        if path not in PAGES:
            self.send_response(404)
            self.end_headers()
            return
        if path not in seen:
            seen.add(path)
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.end_headers()
            return
        body = PAGES[path].encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"сервер: {fmt % args}", file=sys.stderr)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    server = HTTPServer(("127.0.0.1", 8777), Handler)
    print("слухаю http://127.0.0.1:8777/ (Ctrl+C щоб зупинити)")
    server.serve_forever()


if __name__ == "__main__":
    main()
