"""
EC2 SSH Tunnel Manager
======================
Quản lý SSH SOCKS5 tunnel tới EC2 proxy.
App tự động start/stop tunnel — không cần thao tác thủ công.

Cách hoạt động:
- Start SSH process: ssh -D 1080 -N ec2-user@<ELASTIC_IP>
- Background thread monitor health mỗi 15s
- Tự restart nếu tunnel drop
- Tự stop khi app đóng
"""

import atexit
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class EC2TunnelManager:
    """
    Manages SSH SOCKS5 tunnel to EC2 proxy instance.

    Usage:
        manager = EC2TunnelManager()
        manager.start()          # App startup
        manager.stop()           # App shutdown
        manager.wait_ready()     # Block until tunnel is up

    Or as context manager:
        with EC2TunnelManager() as manager:
            ...  # tunnel is up during this block
    """

    def __init__(
        self,
        host: str | None = None,
        key_path: str | None = None,
        user: str | None = None,
        local_port: int | None = None,
        health_check_interval: int = 15,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 999,
    ):
        self.host = host or os.getenv("EC2_PROXY_HOST", "")
        self.key_path = key_path or os.getenv("EC2_PROXY_KEY_PATH", "")
        self.user = user or os.getenv("EC2_PROXY_USER", "ec2-user")
        self.local_port = local_port or int(os.getenv("EC2_PROXY_PORT", "1080"))
        self.health_check_interval = health_check_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._process: subprocess.Popen | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False

        # Register cleanup on exit
        atexit.register(self.stop)

    @property
    def enabled(self) -> bool:
        """EC2 proxy có được bật không (từ .env)."""
        return os.getenv("EC2_PROXY_ENABLED", "false").lower() == "true"

    @property
    def proxy_url(self) -> str:
        return f"socks5h://127.0.0.1:{self.local_port}"

    def _validate_config(self) -> bool:
        """Kiểm tra config đầy đủ chưa."""
        missing = []
        if not self.host:
            missing.append("EC2_PROXY_HOST")
        if not self.key_path:
            missing.append("EC2_PROXY_KEY_PATH")
        if not Path(self.key_path).exists() if self.key_path else True:
            missing.append(f"Key file not found: {self.key_path}")

        if missing:
            log.error("EC2 Proxy config missing: %s", ", ".join(missing))
            log.error("Run: python tools/ec2_proxy/deploy_ec2.py")
            return False
        return True

    def _is_port_listening(self) -> bool:
        """Kiểm tra SOCKS5 port có đang listen không."""
        try:
            with socket.create_connection(("127.0.0.1", self.local_port), timeout=2):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    def _build_ssh_command(self) -> list[str]:
        """Build SSH tunnel command."""
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "TCPKeepAlive=yes",
            "-o",
            "LogLevel=ERROR",
            "-i",
            str(self.key_path),
            "-D",
            str(self.local_port),
            "-N",  # No remote command
            f"{self.user}@{self.host}",
        ]
        return cmd

    def _start_ssh_process(self) -> bool:
        """Start SSH subprocess. Returns True if started successfully."""
        with self._lock:
            # Kill existing process
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()

            try:
                self._process = subprocess.Popen(
                    self._build_ssh_command(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                log.debug("SSH tunnel process started (PID: %d)", self._process.pid)
                return True
            except FileNotFoundError:
                log.error("SSH not found. Install OpenSSH: Settings → Optional Features → OpenSSH Client")
                return False
            except Exception as e:
                log.error("Failed to start SSH tunnel: %s", e)
                return False

    def _monitor_loop(self) -> None:
        """Background thread: monitor tunnel health and reconnect if needed."""
        attempts = 0

        while not self._stop_event.is_set():
            # Check if SSH process is alive
            process_alive = self._process and self._process.poll() is None

            if not process_alive or not self._is_port_listening():
                if self._ready_event.is_set():
                    log.warning("EC2 tunnel dropped. Reconnecting...")
                    self._ready_event.clear()

                attempts += 1
                if attempts > self.max_reconnect_attempts:
                    log.error("Max reconnect attempts reached. Giving up.")
                    break

                log.info("EC2 tunnel connecting (attempt %d)...", attempts)
                if self._start_ssh_process():
                    # Wait for port to become available
                    for _ in range(20):  # 10 seconds max
                        if self._stop_event.is_set():
                            return
                        if self._is_port_listening():
                            log.info("EC2 tunnel ready ✅ (proxy: %s)", self.proxy_url)
                            self._ready_event.set()
                            attempts = 0
                            break
                        time.sleep(0.5)
                    else:
                        log.warning("Tunnel started but port not listening yet...")
                        time.sleep(self.reconnect_delay)
                else:
                    time.sleep(self.reconnect_delay)
            else:
                # Tunnel is healthy
                self._stop_event.wait(timeout=self.health_check_interval)

    def start(self) -> bool:
        """
        Start the SSH tunnel manager.
        Returns True if enabled and started, False if disabled or failed.
        """
        if not self.enabled:
            log.debug("EC2 proxy disabled (EC2_PROXY_ENABLED != true)")
            return False

        if self._running:
            log.debug("Tunnel manager already running")
            return True

        if not self._validate_config():
            return False

        log.info("Starting EC2 SSH tunnel → %s@%s (SOCKS5 port %d)", self.user, self.host, self.local_port)

        self._stop_event.clear()
        self._ready_event.clear()
        self._running = True

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="EC2TunnelMonitor",
            daemon=True,
        )
        self._monitor_thread.start()
        return True

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """
        Block until tunnel is up or timeout.
        Returns True if tunnel is ready, False if timeout.
        """
        if not self.enabled:
            return True  # Not needed, proceed normally

        ready = self._ready_event.wait(timeout=timeout)
        if not ready:
            log.error("EC2 tunnel not ready after %.0fs. Check connectivity.", timeout)
        return ready

    def stop(self) -> None:
        """Stop the tunnel manager and SSH process."""
        if not self._running:
            return

        log.info("Stopping EC2 tunnel...")
        self._running = False
        self._stop_event.set()
        self._ready_event.clear()

        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        log.info("EC2 tunnel stopped.")

    def get_status(self) -> dict:
        """Return current tunnel status."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "ready": self._ready_event.is_set(),
            "host": self.host,
            "port": self.local_port,
            "proxy_url": self.proxy_url if self.enabled else None,
            "port_listening": self._is_port_listening() if self.enabled else None,
            "process_pid": self._process.pid if self._process and self._process.poll() is None else None,
        }

    def __enter__(self):
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *args):
        self.stop()


# Singleton — dùng chung toàn app
_tunnel_manager: EC2TunnelManager | None = None


def get_tunnel_manager() -> EC2TunnelManager:
    """Get singleton tunnel manager instance."""
    global _tunnel_manager
    if _tunnel_manager is None:
        _tunnel_manager = EC2TunnelManager()
    return _tunnel_manager
