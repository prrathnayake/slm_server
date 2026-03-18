#!/usr/bin/env python3
"""
SLM Platform Launcher
Starts all platform services without Docker.
Usage: python run_platform.py [--no-gui] [--gateway-only] [--kill]
"""

import subprocess
import sys
import time
import socket
import threading
import argparse
import os
import signal

SERVICES = []
RUNNING = True


REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "httpx": "httpx",
    "pydantic": "pydantic",
}

PLATFORM_PORTS = {
    "Gateway API": 8000,
    "Runtime Manager": 8001,
    "Trainer Worker": 8002,
}


def check_dependencies():
    missing = []
    for name, import_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Run: python -m pip install {' '.join(missing)}")
        return False
    return True


def print_banner():
    print("\n" + "=" * 55)
    print("       SLM Platform Launcher v0.1.0")
    print("=" * 55)


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port_processes():
    """Kill any processes using platform ports."""
    print("[CLEANUP] Checking for leftover processes...")
    killed = 0
    for name, port in PLATFORM_PORTS.items():
        if check_port(port):
            print(f"  [KILL] Port {port} ({name}) in use - killing process...")
            if sys.platform == "win32":
                # Find and kill process on Windows
                try:
                    result = subprocess.run(
                        f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{port}\') do @echo %a',
                        shell=True, capture_output=True, text=True
                    )
                    pids = set(result.stdout.strip().split())
                    for pid in pids:
                        if pid and pid.isdigit():
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                            killed += 1
                except Exception:
                    pass
            else:
                try:
                    result = subprocess.run(
                        f"lsof -ti:{port}", shell=True, capture_output=True, text=True
                    )
                    for pid in result.stdout.strip().split():
                        if pid:
                            os.kill(int(pid), signal.SIGKILL)
                            killed += 1
                except Exception:
                    pass
    if killed:
        time.sleep(1)
        print(f"  Killed {killed} process(es)")
    else:
        print("  No leftover processes found")


def stream_output(name, proc):
    while RUNNING and proc and proc.poll() is None:
        try:
            line = proc.stdout.readline()
            if line:
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    print(f"[{name}] {decoded}")
        except Exception:
            break


def start_service(name, cmd, port=None):
    if port and check_port(port):
        print(f"[SKIP] {name} - port {port} already in use")
        return None

    print(f"[START] {name}...")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    SERVICES.append({"name": name, "proc": proc, "port": port, "pid": proc.pid})

    t = threading.Thread(target=stream_output, args=(name, proc), daemon=True)
    t.start()

    return proc


def stop_all():
    global RUNNING
    RUNNING = False

    print("\n\n[SHUTDOWN] Stopping all services...\n")

    for svc in SERVICES:
        proc = svc["proc"]
        name = svc["name"]
        pid = svc.get("pid")

        if proc and proc.poll() is None:
            print(f"  [STOP] {name} (pid {pid})")

            if sys.platform == "win32":
                # Kill process tree on Windows
                try:
                    subprocess.run(
                        f"taskkill /F /T /PID {pid}",
                        shell=True,
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            else:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception:
                    pass

    # Double-check ports are free
    time.sleep(1)
    for name, port in PLATFORM_PORTS.items():
        if check_port(port):
            print(f"  [WARN] Port {port} still in use - force killing...")
            kill_port_processes()
            break

    print("\n[DONE] All services stopped.\n")


def check_health(port, path="/health", timeout=10):
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def launch_ui():
    print("[START] Desktop UI...")
    proc = subprocess.Popen(
        "python -m local_llm_platform.apps.desktop.app",
        shell=True,
    )
    SERVICES.append({"name": "Desktop UI", "proc": proc, "port": None, "pid": proc.pid})


def main():
    parser = argparse.ArgumentParser(description="SLM Platform Launcher")
    parser.add_argument("--no-gui", action="store_true", help="Don't launch desktop UI")
    parser.add_argument("--gateway-only", action="store_true", help="Only start gateway API")
    parser.add_argument("--kill", action="store_true", help="Kill leftover processes and exit")
    args = parser.parse_args()

    print_banner()

    # Kill mode
    if args.kill:
        kill_port_processes()
        return

    if not check_dependencies():
        sys.exit(1)

    # Check ports
    any_busy = False
    for name, port in PLATFORM_PORTS.items():
        if check_port(port):
            print(f"  [WARN] Port {port} ({name}) is already in use")
            any_busy = True

    if any_busy:
        print("\n  Kill existing processes? (y/n): ", end="")
        if input().strip().lower() == "y":
            kill_port_processes()
        else:
            print("Aborted.")
            return

    print("\n" + "-" * 55)

    # Start services
    start_service("Gateway API", "python -m local_llm_platform.apps.gateway_api.main", 8000)
    time.sleep(2)

    if not args.gateway_only:
        start_service("Runtime Manager", "python -m local_llm_platform.apps.runtime_manager.main", 8001)
        time.sleep(1)
        start_service("Trainer Worker", "python -m local_llm_platform.apps.trainer_worker.main", 8002)
        time.sleep(1)

    # Health check
    print("\n[WAIT] Checking Gateway API health...")
    if check_health(8000):
        print("[OK] Gateway API is ready!")
    else:
        print("[WARN] Gateway API not responding yet, continuing...")

    # Status
    print("\n" + "=" * 55)
    print("  PLATFORM RUNNING")
    print("=" * 55)
    print("  Gateway API:      http://127.0.0.1:8000")
    print("  API Docs:         http://127.0.0.1:8000/docs")
    if not args.gateway_only:
        print("  Runtime Manager:  http://127.0.0.1:8001")
        print("  Trainer Worker:   http://127.0.0.1:8002")
    print("=" * 55)

    if not args.no_gui:
        time.sleep(2)
        launch_ui()
        print("  Desktop UI:       Launched in separate window")
    else:
        print("  Desktop UI:       Disabled (--no-gui)")

    print("=" * 55)
    print("  Ctrl+C to stop all services")
    print("=" * 55 + "\n")

    # Monitor
    try:
        while True:
            for svc in SERVICES[:]:  # copy to avoid mutation
                proc = svc["proc"]
                if proc and proc.poll() is not None and svc["name"] != "Desktop UI":
                    print(f"\n[ERROR] {svc['name']} exited with code {proc.returncode}")
                    stop_all()
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
