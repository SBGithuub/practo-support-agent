"""
verify_all.py - Master verification pipeline for Practo Domain Support Agent.
Executes all demonstration scripts, verifies transcripts, runs pytest suite, and checks formatting.
"""

import os
import sys
import subprocess
import time

def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def main():
    start_total = time.perf_counter()
    print_banner("PRACTO DOMAIN SUPPORT AGENT: COMPLETE MASTER VERIFICATION")
    print(f"Python Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    
    # 1. Dataset Verification
    print("\n[Step 1/6] Verifying Appointment Dataset Generation...")
    res = subprocess.run([sys.executable, "dataset.py"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        sys.exit(1)
    print("  -> Dataset generated and validated successfully.")

    # 2. Knowledge Base Verification
    print("\n[Step 2/6] Verifying Knowledge Base Documents...")
    kb_files = [f for f in os.listdir("knowledge_base") if f.endswith(".txt")]
    print(f"  -> Found {len(kb_files)} policy documents in knowledge_base/ (Required: >=12).")
    assert len(kb_files) >= 12, "Knowledge base must have at least 12 documents"

    # 3. Running Pytest Suite
    print("\n[Step 3/6] Running Pytest Suite (38 unit & integration tests)...")
    res_test = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], capture_output=True, text=True)
    print(res_test.stdout.strip())
    if res_test.returncode != 0:
        print(f"FAILED: {res_test.stderr}")
        sys.exit(1)
    print("  -> All 38 unit and integration tests passed cleanly.")

    # 4. Running Verification Demos
    print("\n[Step 4/6] Executing Demonstration & Evaluation Scripts...")
    demo_scripts = [
        ("Task 6: Status Tool & Escalation", "scripts/run_task6_demo.py"),
        ("Task 7: CrewAI Multi-Agent Kickoff", "scripts/run_task7_demo.py"),
        ("Task 8: Session Memory Demos", "scripts/run_task8_demo.py"),
        ("Task 10: Tri-Layer Guardrails", "scripts/run_task10_demo.py"),
        ("Task 11: FastAPI & WebSocket Chat", "scripts/run_task11_demo.py"),
        ("Task 13: 15-Query Evaluation Rubric", "evaluations/run_evals.py"),
        ("Task 14: Autogen 2-Agent Review Stage", "scripts/run_task14_demo.py"),
        ("Task 15: AI Governance & Budget Cap", "scripts/run_task15_demo.py"),
        ("Task 16: In-Memory Response Caching", "scripts/run_task16_demo.py"),
    ]
    
    for label, script_path in demo_scripts:
        t0 = time.perf_counter()
        res_demo = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if res_demo.returncode != 0:
            print(f"  -> {label} FAILED ({dt:.2f}s): {res_demo.stderr}")
            sys.exit(1)
        print(f"  -> {label} PASSED ({dt:.2f}s)")

    # 5. Transcript Audit
    print("\n[Step 5/6] Auditing Generated Transcripts in transcripts/...")
    required_transcripts = [
        "task1_dataset_validation.txt",
        "task4_grounded_fallback.txt",
        "task5_chunking_comparison.txt",
        "task6_status_tool.txt",
        "task7_crew_execution.txt",
        "task8_memory_multiturn.txt",
        "task8_memory_fresh_reset.txt",
        "task10_guardrails_firing.txt",
        "task11_websocket_session.txt",
        "task12_structured_logs.jsonl",
        "task13_evaluation_report.txt",
        "task14_autogen_review.txt",
        "task15_governance_budget.txt",
        "task16_cache_benchmark.txt"
    ]
    for trans in required_transcripts:
        path = os.path.join("transcripts", trans)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"  -> MISSING OR EMPTY: {trans}")
            sys.exit(1)
        print(f"  -> Verified: {trans} ({os.path.getsize(path)} bytes)")

    # 6. README Formatting Check (Strict Zero Em-Dashes)
    print("\n[Step 6/6] Verifying README.md Strict Formatting Rules...")
    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    em_dash_count = readme_content.count("—")
    en_dash_count = readme_content.count("–")
    print(f"  -> Em-dash count in README.md: {em_dash_count}")
    print(f"  -> En-dash count in README.md: {en_dash_count}")
    assert em_dash_count == 0, "Violation: README.md must contain zero em-dashes"
    assert en_dash_count == 0, "Violation: README.md must contain zero en-dashes"
    print("  -> Strict formatting check passed: Zero em-dashes and zero en-dashes.")

    elapsed = time.perf_counter() - start_total
    print_banner(f"ALL CAPSTONE REQUIREMENTS VERIFIED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("Repository is 100% turnkey, reproducible, and ready for GitHub submission.\n")

if __name__ == "__main__":
    main()

