"""
Port Management Utilities

Helper functions for port availability checking and management.
"""

import socket
from typing import Optional


def is_port_available(port: int, host: str = '127.0.0.1') -> bool:
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host address (default: localhost)

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(
    start_port: int = 5003,
    max_tries: int = 10,
    host: str = '127.0.0.1'
) -> Optional[int]:
    """
    Find an available port starting from start_port.

    Iterates through port numbers starting from start_port until
    an available port is found or max_tries is reached.

    Args:
        start_port: Starting port number to try
        max_tries: Maximum number of ports to try
        host: Host address to bind to

    Returns:
        Available port number if found, None otherwise

    Raises:
        RuntimeError: If no available port found within range
    """
    for port in range(start_port, start_port + max_tries):
        if is_port_available(port, host):
            return port

    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_tries - 1}"
    )


def kill_process_on_port(port: int) -> bool:
    """
    Attempt to kill process using the specified port.

    Note: This is a placeholder. Implementation would require
    platform-specific code using psutil or similar.

    Args:
        port: Port number

    Returns:
        True if successful, False otherwise
    """
    # This would require psutil or platform-specific code
    # For now, return False to indicate not implemented
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"kill_process_on_port({port}) not implemented")
    return False
