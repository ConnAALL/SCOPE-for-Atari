"""
Script for setting up a Ray cluster on a remote server
"""

import argparse
import os
import shlex
import socket
import subprocess
import time


def ssh_command(username: str,
                password: str,
                hostname: str,
                command: str,
                timeout: int = 10) -> tuple[int, str, str]:
    """
    Runs `command` on `hostname` via the system SSH client, waits until it's done (or times out).
    Returns (exit_code, stdout_str, stderr_str).
    This version uses `sshpass` to pass the password, and skips host-key checks.
    """
    # NOTE: you must have `sshpass` installed on your system for this to work
    ssh_opts = "-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null"
    ssh_cmd = (
        f"sshpass -p {shlex.quote(password)} ssh {ssh_opts} "
        f"{username}@{hostname} {shlex.quote(command)}"
    )

    try:
        proc = subprocess.Popen(
            shlex.split(ssh_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            out_bytes, err_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out_bytes, err_bytes = proc.communicate()
            return -1, "", f"SSH command timed out after {timeout} seconds"

        exit_code = proc.returncode
        out = out_bytes.decode("utf-8", errors="ignore")
        err = err_bytes.decode("utf-8", errors="ignore")
        return exit_code, out, err

    except Exception as e:
        raise RuntimeError(f"Failed to run SSH command: {e}")


def wait_for_port_open(host: str,
                       port: int,
                       timeout_secs: int = 15) -> bool:
    """
    Simple TCP-connect loop to check if `host:port` is reachable or not
    Returns True if it can connect within timeout_secs, False otherwise
    """
    start = time.time()
    print(f"[ray_setup] Waiting for {host}:{port} to become reachable…")
    # For timeout_secs seconds, try to establish a connection to the host on the given port
    while time.time() - start < timeout_secs:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            time.sleep(1)
    return False


def setup_ray_cluster(username: str,
                      password: str,
                      head_host: str,
                      head_ip: str,
                      worker_hosts: list[str]):
    """
    Setup a Ray cluster by starting the Ray head on the head host and then starting the Ray workers on the worker hosts.
    """
    # Stop any existing Ray cluster
    os.system("ray stop --force")

    # Start the Ray head
    print(f"[ray_setup] Starting Ray head on {head_host} (IP={head_ip})...")
    head_command = f"ray start --include-dashboard=True --head --port=6379 --num-cpus=48 --num-gpus=0"
    os.system(head_command)
    print(f"[ray_setup] Head started successfully")

    # Wait until port 6379 is truly open on the head
    print(f"[>] Waiting for {head_ip}:6379 to become reachable…")
    if not wait_for_port_open(head_ip, 6379, timeout_secs=15):
        print("[ray_setup] Timeout: head node never opened port 6379. Check firewall or `ray start` logs.")
        return
    print("[ray_setup] head is listening on 6379\n")

    # For each worker, stop the existing Ray cluster and start a new one
    for worker in worker_hosts:
        worker_ip = worker
        worker_cmd = (
            f"ray stop --force; "
            f"ray start "
            f"--address='{head_ip}:6379' "
        )

        print(f"[ray_setup] Starting Ray worker on {worker} (IP={worker_ip})…")
        code, out, err = ssh_command(username, password, worker, worker_cmd)  # Run the worker command on the worker host
        
        if code != 0: print(f"[ray_setup] ERROR launching worker on {worker}:\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        else: print(f"[ray_setup] worker {worker} launched, stdout:\n{out.strip()}\n")

    print("[ray_setup] All SSH commands issued. Give Ray a few seconds to form the cluster.\n")


def main():
    # Argument parser for the username and password
    parser = argparse.ArgumentParser(description='Set up a Ray cluster on remote servers')
    parser.add_argument('--username', required=True, help='Username for the remote servers')
    parser.add_argument('--password', required=True, help='Password for the remote servers')
    args = parser.parse_args()

    # Head host is the server that will be used to launch the Ray head
    # Head IP is the IP address of the head host
    head_host = "136.244.224.97"
    head_ip   = "136.244.224.97"

    # Worker hosts are the servers that will be used to launch the Ray workers
    worker_hosts = [
        "136.244.224.187",
        "136.244.224.200",
        "136.244.224.230",
        "136.244.224.90",
        "136.244.224.242",
        "136.244.224.107",
        "136.244.224.27",
        "136.244.224.91",
        "136.244.224.101",
        "136.244.224.108",
        "136.244.224.35",
        "136.244.224.32",     
    ]

    # Setup the Ray cluster
    setup_ray_cluster(
        username=args.username,
        password=args.password,
        head_host=head_host,
        head_ip=head_ip,
        worker_hosts=worker_hosts
    )


if __name__ == "__main__":
    main()
