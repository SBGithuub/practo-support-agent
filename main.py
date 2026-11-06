"""
main.py - Interactive CLI and central entry point for the Practo Domain Support Agent.
Allows non-technical users and evaluators to easily run, test, and interact with the project.
"""

import os
import sys
import argparse
import subprocess

# Auto-re-route if invoked by default Python 3.14 (or wrong python without uvicorn/crewai)
if sys.version_info >= (3, 13) or sys.version_info < (3, 10):
    py312_candidates = [
        r"C:\Python312\python.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe")
    ]
    py312_exe = None
    for cand in py312_candidates:
        if os.path.exists(cand):
            py312_exe = cand
            break
            
    if not py312_exe:
        try:
            check = subprocess.run(["py", "-3.12", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
            if check.returncode == 0 and check.stdout.strip():
                py312_exe = check.stdout.strip()
        except Exception:
            pass
            
    if py312_exe:
        # Re-invoke script under Python 3.12 environment where packages are installed
        cmd = [py312_exe, os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))

# Set project environment defaults
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

def run_interactive_chat():
    """Starts an interactive multi-turn terminal chat with the support agent."""
    from app.agents.crew import run_practo_support_pipeline
    from app.agents.memory import ConversationalSessionManager
    
    print("\n" + "=" * 75)
    print(" PRACTO DOMAIN SUPPORT AGENT: INTERACTIVE TERMINAL CHAT")
    print("=" * 75)
    print("Mode: Deterministic MOCK_LLM (Zero API Keys, Zero Network Required)")
    print("Capabilities:")
    print("  1. Policy Inquiries (e.g. 'What is the cancellation policy?', 'Consultation fees')")
    print("  2. Appointment Status (e.g. 'Check status for APT-0001' or 'APT-0007')")
    print("  3. Conversational Memory (maintains context across turns within this session)")
    print("Type 'exit' or 'quit' to return to menu.\n" + "-" * 75)

    session_id = "cli_interactive_session"
    
    def _pipeline(query: str, history_context: str) -> str:
        res = run_practo_support_pipeline(query, history_context=history_context)
        return res.answer

    manager = ConversationalSessionManager(_pipeline)

    while True:
        try:
            user_input = input("\nYou (Patient): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive chat session.")
                break
                
            print("Practo Agent is thinking...")
            answer = manager.chat(session_id, user_input)
            print(f"\nPracto Agent:\n{answer}")
        except KeyboardInterrupt:
            print("\nExiting interactive chat.")
            break
        except Exception as e:
            print(f"\nError: {e}")

def start_fastapi_server(host: str = "127.0.0.1", port: int = 8000):
    """Starts the FastAPI backend with HTTP endpoints and WebSocket support."""
    import uvicorn
    print("\n" + "=" * 75)
    print(f" Starting FastAPI Production Server on http://{host}:{port}")
    print("=" * 75)
    print("Available Endpoints:")
    print(f"  - Interactive Swagger Docs: http://{host}:{port}/docs")
    print(f"  - Health Probe:             http://{host}:{port}/health")
    print(f"  - Policy / Status API:      POST http://{host}:{port}/ask")
    print(f"  - Dynamic Indexing API:     POST http://{host}:{port}/add-document")
    print(f"  - Real-Time WebSocket Chat: ws://{host}:{port}/ws/chat")
    print("\nPress CTRL+C to stop the server.\n" + "-" * 75)
    uvicorn.run("app.api.main:app", host=host, port=port, reload=False)

def run_evaluation_benchmark():
    """Runs the 15-query LLM-as-judge evaluation."""
    from evaluations.run_evals import main as eval_main
    eval_main()

def run_tests():
    """Runs the entire 38-test pytest suite."""
    import pytest
    print("\nRunning complete test suite with pytest...")
    pytest.main(["tests/", "-v"])

def run_master_verification():
    """Runs verify_all.py."""
    import subprocess
    subprocess.run([sys.executable, "verify_all.py"])

def display_menu():
    print("\n" + "=" * 70)
    print("       PRACTO DOMAIN SUPPORT AGENT: MAIN DASHBOARD")
    print("=" * 70)
    print("Please choose an option to run:")
    print("  [1] Interactive Terminal Chat (Ask policy or appointment questions)")
    print("  [2] Start FastAPI Web Backend (HTTP + WebSocket server on port 8000)")
    print("  [3] Run Complete Master Verification (Audits all 16 tasks & transcripts)")
    print("  [4] Run Pytest Test Suite (Executes 38 unit and integration tests)")
    print("  [5] Run 15-Query Evaluation Benchmark (LLM-as-Judge scoring)")
    print("  [6] Exit")
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Practo Domain Support Agent Central Runner")
    parser.add_argument("--chat", action="store_true", help="Launch interactive terminal chat")
    parser.add_argument("--server", action="store_true", help="Start FastAPI production server")
    parser.add_argument("--verify", action="store_true", help="Run master verification pipeline")
    parser.add_argument("--test", action="store_true", help="Run pytest suite")
    parser.add_argument("--eval", action="store_true", help="Run 15-query benchmark evaluation")
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    
    args = parser.parse_args()

    if args.chat:
        run_interactive_chat()
    elif args.server:
        start_fastapi_server(host=args.host, port=args.port)
    elif args.verify:
        run_master_verification()
    elif args.test:
        run_tests()
    elif args.eval:
        run_evaluation_benchmark()
    else:
        # If run with no CLI flags, show friendly interactive menu
        while True:
            display_menu()
            choice = input("Enter option (1-6): ").strip()
            if choice == "1":
                run_interactive_chat()
            elif choice == "2":
                start_fastapi_server()
            elif choice == "3":
                run_master_verification()
            elif choice == "4":
                run_tests()
            elif choice == "5":
                run_evaluation_benchmark()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid selection. Please choose a number between 1 and 6.")

if __name__ == "__main__":
    main()

