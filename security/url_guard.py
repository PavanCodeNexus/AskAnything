"""SSRF (Server-Side Request Forgery) protection and URL validation guard."""
import ipaddress
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

from utils.logger import setup_logger

logger = setup_logger("url_guard")

# Blocked hostnames (cloud metadata, internal services)
BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata.internal",
    "instance-data",
}

# Blocked IP networks (loopback, private networks, cloud link-local metadata)
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("10.0.0.0/8"),         # Private RFC1918
    ipaddress.ip_network("100.64.0.0/10"),      # Shared address space
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / Cloud Metadata (AWS, GCP, Azure)
    ipaddress.ip_network("172.16.0.0/12"),      # Private RFC1918
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),     # Private RFC1918
    ipaddress.ip_network("198.18.0.0/15"),      # Network benchmark tests
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6 ranges
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),            # IPv6 Loopback
    ipaddress.ip_network("fc00::/7"),           # Unique local address
    ipaddress.ip_network("fe80::/10"),          # Link-local unicast
    ipaddress.ip_network("ff00::/8"),           # Multicast
]


def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validates URL against SSRF vulnerabilities, ensuring public HTTP/HTTPS access only.

    Args:
        url: User-provided URL string.

    Returns:
        Tuple of (is_safe: bool, reason_if_unsafe: Optional[str])
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {str(e)}"

    # Restrict scheme strictly to http and https
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"Forbidden URL scheme '{parsed.scheme}'. Only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL missing valid hostname."

    hostname_lower = hostname.lower()

    # Direct check against known restricted hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        logger.warning("SSRF blocked hostname: %s", hostname)
        return False, f"Access to '{hostname}' is blocked for security reasons."

    # Prevent access to IP addresses in private/link-local ranges
    try:
        # Resolve all DNS A and AAAA records
        addr_info = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        resolved_ips = {item[4][0] for item in addr_info}
    except socket.gaierror as e:
        logger.warning("Failed to resolve hostname %s: %s", hostname, e)
        return False, f"Could not resolve host '{hostname}'."
    except Exception as e:
        logger.warning("DNS resolution error for %s: %s", hostname, e)
        return False, f"DNS error: {str(e)}"

    if not resolved_ips:
        return False, f"Host '{hostname}' resolved to no IP addresses."

    for ip_str in resolved_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            for blocked_net in BLOCKED_IP_NETWORKS:
                if ip_obj in blocked_net:
                    logger.warning("SSRF blocked attempt to connect to %s (%s)", hostname, ip_str)
                    return False, f"URL resolves to private or restricted IP address: {ip_str}"
        except ValueError:
            return False, f"Invalid IP address representation: {ip_str}"

    return True, None
