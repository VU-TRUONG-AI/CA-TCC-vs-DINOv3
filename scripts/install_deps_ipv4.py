"""Use IPv4 for pip in this process only when local IPv6 downloads stall."""
import runpy
import socket
import sys

_original = socket.getaddrinfo

def ipv4_address(host, port, family=0, type=0, proto=0, flags=0):
    return _original(host, port, socket.AF_INET if family == socket.AF_UNSPEC else family, type, proto, flags)

socket.getaddrinfo = ipv4_address
sys.argv = ['pip', *sys.argv[1:]]
runpy.run_module('pip', run_name='__main__')
