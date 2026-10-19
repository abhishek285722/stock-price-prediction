"""
launch.py - Dedicated Application Window Launcher (Stock Price Prediction)
Starts the Streamlit server and launches the app in an isolated application window.
When Stop Session is clicked in the web app, the browser window closes automatically.
If the browser window is closed, the server terminates automatically.
"""

import os
import sys
import time
import socket
import tempfile
import urllib.request
import subprocess


def find_available_port(start_port=8504):
    """Finds an available TCP port starting from start_port."""
    port = start_port
    while port < start_port + 20:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def wait_for_server(port, timeout=15):
    """Polls the server until it responds to HTTP requests."""
    url = f"http://localhost:{port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def get_browser_exe():
    """Detects Edge or Chrome executable path on Windows."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_script = os.path.join(base_dir, "app.py")
    port = find_available_port(8504)
    url = f"http://localhost:{port}"

    print(f"[Starting Streamlit server on port {port}...]")

    # Start Streamlit server in headless mode
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_script,
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    server_process = subprocess.Popen(cmd, cwd=base_dir)

    # Wait until server is live
    wait_for_server(port)

    # Launch browser in isolated app window
    browser_exe = get_browser_exe()
    profile_dir = os.path.join(tempfile.gettempdir(), f"stock_win_profile_{port}")

    browser_proc = None
    if browser_exe:
        print("[Launching isolated application window...]")
        browser_cmd = [
            browser_exe,
            f"--app={url}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        browser_proc = subprocess.Popen(browser_cmd)
    else:
        print("[Opening default web browser...]")
        import webbrowser
        webbrowser.open_new(url)

    # Monitor both processes:
    # 1. If user clicks "Stop Session" in web app -> server_process exits -> kill browser_proc so window closes immediately!
    # 2. If user closes window with 'X' -> browser_proc exits -> terminate server_process!
    try:
        while True:
            # Check if server exited (Stop Session clicked in web app)
            if server_process.poll() is not None:
                if browser_proc and browser_proc.poll() is None:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(browser_proc.pid)],
                            capture_output=True,
                        )
                    except Exception:
                        browser_proc.kill()
                break

            # Check if user closed the window with 'X'
            if browser_proc and browser_proc.poll() is not None:
                try:
                    server_process.terminate()
                except Exception:
                    pass
                break

            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if server_process.poll() is None:
            server_process.kill()
        if browser_proc and browser_proc.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(browser_proc.pid)],
                    capture_output=True,
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
