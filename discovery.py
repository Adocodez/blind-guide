"""
discovery.py

Shared mDNS constants + helpers so the server can advertise itself and the
client can find it without either side needing to know the other's IP ahead
of time. Solves the "hotspot reassigns a new IP every time it's toggled"
problem — the client just asks the network "where is blindguide?" instead
of being told an address.

Install (both server and client machines): pip install zeroconf
"""

import socket

from zeroconf import Zeroconf

SERVICE_TYPE = "_blindguide._tcp.local."
SERVICE_NAME = "blindguide." + SERVICE_TYPE
HOSTNAME = "blindguide"

# Must match whatever port you run `uvicorn server:app --port <this>` with.
DEFAULT_PORT = 8000


def get_local_ip() -> str:
    """
    Returns the IP of the network interface that would be used to reach the
    LAN/internet — i.e. the address other devices on the same hotspot should
    use to reach this machine. Doesn't actually send traffic; a UDP
    connect() just makes the OS pick the right local interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def resolve_server(zc: Zeroconf, timeout_s: float = 5.0) -> tuple[str, int] | None:
    """
    Client-side: actively queries the network for the blindguide service and
    returns (ip, port), or None if nothing answered within timeout_s. This
    does the mDNS query itself rather than relying on the OS to resolve
    "blindguide.local" — more reliable across Windows/mobile than trusting
    every platform's mDNS resolver to handle arbitrary .local hostnames.
    """
    info = zc.get_service_info(SERVICE_TYPE, SERVICE_NAME, timeout=int(timeout_s * 1000))
    if info is None or not info.addresses:
        return None
    ip = socket.inet_ntoa(info.addresses[0])
    return ip, info.port
