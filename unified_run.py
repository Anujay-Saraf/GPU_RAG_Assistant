import sys
import subprocess
import argparse
import signal
import time

def start_backend():
    print("🚀 Starting FastAPI Gateway on http://127.0.0.1:8000 ...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=None,
        stderr=None
    )

def start_frontend():
    print("⚡ Starting Streamlit Portal on http://127.0.0.1:8501 ...")
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "frontend/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ],
        stdout=None,
        stderr=None
    )

def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Service Orchestrator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--backend", action="store_true", help="Run only the FastAPI Backend Gateway")
    group.add_argument("--frontend", action="store_true", help="Run only the Streamlit Frontend UI")
    args = parser.parse_args()

    processes = []

    if args.backend:
        processes.append(start_backend())
    elif args.frontend:
        processes.append(start_frontend())
    else:
        # Default: Run both services concurrently
        processes.append(start_backend())
        time.sleep(2)  # Give backend a 2-second lead to initialize database connections
        processes.append(start_frontend())

    print("\n✅ Service(s) running. Press CTRL+C to terminate all services cleanly.\n")

    def shutdown(sig, frame):
        print("\n🛑 Terminating all services...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main orchestrator process alive
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()