#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import http.server
from http.server import SimpleHTTPRequestHandler as HTTPHandler

"""
A simple http server (python3)
"""
class CustomHttpRequestHandler(HTTPHandler):
    def do_GET(self):
        if (self.path == "/"):
            self.path = "index.html"

        elif (self.path == "/favicon.ico"):
            self.path = "img/favicon.ico"

        return HTTPHandler.do_GET(self)

port = 8080 if not sys.argv[1:] \
            else int(sys.argv[1])

server_addr = ("127.0.0.1", port)
handler_class = CustomHttpRequestHandler
handler_class.protocol_version = "HTTP/1.0"

httpd = http.server.HTTPServer(server_addr, handler_class)

# print server info
sock_addr = httpd.socket.getsockname()
print(f"Serving HTTP server on {sock_addr[0]}:{sock_addr[1]}")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("end server")