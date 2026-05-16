#!/usr/bin/env python3
"""
Automated LLM Inference Experiment Runner
==========================================
Runs all 80 inference combinations (2 models x 10 reports x 4 prompt types)
through the Ollama API with explicit context clearing between every run.
CAF-2024-001 and CAF-2024-002 = real anonymized safety reports.

Captures inference metrics (TPS, latency, token counts) directly from the
Ollama API and system metrics (CPU utilisation, RAM usage) by sampling
/proc/stat and /proc/meminfo in a background thread for the duration of
each run.

Eliminates context bleed risk and manual handling errors.

Usage:
    python3 run_experiment.py

Output:
    /tmp/experiment_results/results.csv     - all metrics
    /tmp/experiment_results/responses/      - individual response text files
    /tmp/experiment_results/experiment.log  - run log

Requirements:
    - Ollama running on localhost:11434
    - LLaMA 3.1 (llama3.1:latest) and Qwen 2.5 32B (qwen2.5:32b) pulled
    - Python 3 with standard library only (no external dependencies)
    - Linux host (reads from /proc; the rest of the script is portable)

Author: Inusah Mohammed
Karelia University of Applied Sciences
Joensuu, Finland
LinkedIn: https://www.linkedin.com/in/inusah-mohammed/
"""

import json
import urllib.request
import urllib.error
import csv
import os
import sys
import time
import threading
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = "/tmp/experiment_results"
RESPONSE_DIR = f"{OUTPUT_DIR}/responses"
RESULTS_CSV = f"{OUTPUT_DIR}/results.csv"
LOG_FILE    = f"{OUTPUT_DIR}/experiment.log"

SAMPLE_INTERVAL = 1.0  # seconds between system metric samples during a run

MODELS = [
    "llama3.1:latest",
    "qwen2.5:32b",
]

PROMPT_TYPES = {
    "Summarization": (
        "Summarize the following manufacturing safety incident report in 3 to 5 sentences. "
        "Your summary should identify the incident type, severity level, immediate cause, and outcome. "
        "Base your response only on the information provided in the report."
    ),
    "Risk Classification": (
        "Classify the risk level of the following manufacturing safety incident. "
        "Your classification should identify the severity of the incident and the likelihood of recurrence, "
        "and provide a brief justification using standard risk assessment criteria. "
        "Base your response only on the information provided in the report."
    ),
    "Corrective Action Extraction": (
        "Extract and list all corrective actions described in the following manufacturing safety incident report. "
        "Present each corrective action as a separate numbered item. "
        "Do not add corrective actions that are not explicitly stated in the report. "
        "Base your response only on the information provided."
    ),
    "Root Cause Identification": (
        "Identify and explain the root causes of the manufacturing safety incident described in the following report. "
        "For each root cause identified, provide a brief explanation of how it contributed to the incident. "
        "Base your response only on the information provided in the report."
    ),
}

# Embedded reports — keeps the script self-contained and reproducible
REPORTS = {
    "INC-2024-047": (
        "Report ID: INC-2024-047. A hydraulic press operator noticed an unexpected pressure spike reaching approximately 340 bar, "
        "significantly above the standard operating threshold of 280 bar. The operator immediately initiated an emergency stop "
        "before any component failure occurred. A partial blockage was identified in the hydraulic return line caused by accumulated "
        "metal shavings. The inline filter had exceeded its 500-hour service interval at 623 hours. No injuries were sustained and "
        "production was halted for 47 minutes. Corrective actions: replace all hydraulic inline filters on Assembly Line B, revise "
        "pre-shift equipment checklist to include hydraulic pressure baseline check, review and update hydraulic filter replacement "
        "schedule across all production lines, conduct toolbox talk on hydraulic system hazard awareness for all line operators."
    ),
    "INC-2024-051": (
        "Report ID: INC-2024-051. A chemical handling technician sustained minor skin irritation on both forearms while transferring "
        "industrial solvent from a bulk storage drum to smaller dispensing containers. The technician was wearing standard nitrile "
        "gloves but had not donned the required chemical-resistant extended-cuff gloves. A small quantity of solvent splashed onto "
        "the forearms above the glove line. The chemical-resistant gloves were not available at the workstation as stock had been "
        "depleted and not replenished. The technician proceeded with the task using available PPE rather than stopping work. No "
        "minimum stock level alert exists for PPE at chemical handling workstations. First aid was administered on site and no "
        "further medical treatment was required. Corrective actions: restock chemical-resistant extended-cuff gloves at all chemical "
        "handling workstations, implement minimum stock level system for PPE at chemical handling areas, conduct refresher training "
        "on stop-work authority and PPE selection for chemical handling staff, review and update PPE requirements signage at "
        "chemical storage workstations."
    ),
    "INC-2024-058": (
        "Report ID: INC-2024-058. A forklift operator sustained a sprained left wrist and minor lacerations to the left hand while "
        "attempting to manually reposition a misaligned pallet on forklift forks. The operator had lowered the forks and exited the "
        "cab without securing the forklift or obtaining assistance. While pulling the pallet the operator lost footing on a wet "
        "floor surface near the loading bay entrance and fell forward striking the left hand and wrist against the pallet edge. "
        "The operator was unable to return to work for four days. The wet floor was caused by rainwater tracked in from outside "
        "and no absorbent matting or warning signs were in place. Root causes: operator unaware that manual pallet adjustment "
        "without assistance was a procedural violation, no wet floor warning system or absorbent matting at loading bay entrance, "
        "pre-shift briefing did not address weather-related slip hazards. Corrective actions: install permanent absorbent matting "
        "at all loading bay entrances, revise forklift operator training to cover manual handling restrictions, add weather-related "
        "slip hazard checks to pre-shift briefing checklist, conduct toolbox talk on slip hazard awareness for all warehouse staff."
    ),
    "INC-2025-003": (
        "Report ID: INC-2025-003. A CNC machinist opened the machine guard door on a CNC milling machine to inspect a workpiece "
        "mid-cycle without initiating a full machine stop. The machine had paused on a tool change cycle and the machinist believed "
        "the spindle was stationary. However the spindle was still in a deceleration phase rotating at approximately 200 RPM. "
        "The machinist's left hand came within approximately 8 centimetres of the decelerating spindle before the guard interlock "
        "triggered an emergency stop. No contact occurred. Classified as critical due to potential for severe crush or degloving "
        "injury. Root causes: operator training did not cover distinction between tool change pause and full spindle stop, "
        "production time pressure from a backlog contributed to rushed behaviour. Corrective actions: revise CNC operator training "
        "to cover spindle deceleration hazards, add spindle stop confirmation step to mid-cycle inspection procedure, review "
        "production scheduling to address time pressure factors, conduct machine-specific safety briefing for all CNC machinists, "
        "assess feasibility of audible spindle stop confirmation signal."
    ),
    "INC-2025-009": (
        "Report ID: INC-2025-009. A spray painter reported symptoms of mild dizziness and headache approximately 40 minutes into "
        "a scheduled spray painting session in Spray Booth 2. One of the two extract ventilation fans had failed, reducing air "
        "extraction capacity by approximately 50 percent. The painter was wearing a half-face respirator with organic vapour "
        "cartridges as required but the increased vapour concentration combined with physical exertion resulted in mild symptoms. "
        "The fan failure was not detected prior to commencement because the pre-use check relies on operator observation rather "
        "than quantitative measurement. The planned preventive maintenance schedule for the ventilation fans was overdue by three "
        "weeks. Corrective actions: install airflow monitoring sensors with alarm in all spray booths, update pre-use spray booth "
        "checklist to include quantitative airflow verification, review and prioritize overdue preventive maintenance for "
        "ventilation systems, assess feasibility of automatic spray booth lockout on low airflow."
    ),
    "INC-2025-014": (
        "Report ID: INC-2025-014. An electrical maintenance technician received a mild electric shock to the right hand while "
        "working on a control panel reported as de-energised and locked out. The lockout had been applied to the main isolator "
        "but a secondary 24V DC control circuit fed from a separate power supply remained energised. The technician made contact "
        "with a live terminal on the secondary circuit. Classified as critical due to failure of the lockout/tagout procedure to "
        "identify all energy sources. Root causes: panel wiring diagram was outdated and did not reflect a secondary power supply "
        "added two years prior, lockout/tagout procedure does not require live circuit verification test after lockout application, "
        "secondary power supply unit was not labelled on panel exterior. Corrective actions: audit and update all control panel "
        "wiring diagrams, revise lockout/tagout procedure to require live circuit verification test, label all secondary power "
        "supply units on panel exterior, conduct lockout/tagout refresher training for all electrical maintenance staff, review "
        "recent panel modifications for undocumented circuit additions."
    ),
    "INC-2025-028": (
        "Report ID: INC-2025-028. A production operative sustained a laceration approximately 4 centimetres in length to the index "
        "finger of the right hand while attempting to clear a product jam on a moving conveyor belt without first stopping the "
        "conveyor. The operative's finger came into contact with the moving belt edge resulting in a cut that required closure "
        "with three adhesive strips. The operative returned to modified duties the same day. Root causes: operative not fully "
        "aware that stopping the conveyor was mandatory before clearing jams, emergency stop buttons at Section 3 positioned only "
        "at the ends of the conveyor section approximately 3 metres from where the jam occurred. Corrective actions: install "
        "additional emergency stop buttons at 1.5 metre intervals along all conveyor sections, conduct mandatory conveyor safety "
        "retraining for all production operatives, update conveyor jam clearance procedure to include mandatory stop verification "
        "step, display conveyor stop-before-clearing instruction at all jam-prone sections."
    ),
    "INC-2025-035": (
        "Report ID: INC-2025-035. A maintenance technician was using a bench grinder to sharpen a cutting tool when the grinding "
        "wheel fractured and ejected fragments at high velocity. The technician was wearing safety glasses but not the full face "
        "shield specified for grinding operations. A fragment struck the safety glasses lens cracking it but not penetrating. "
        "No injury was sustained. Post-incident inspection revealed the grinding wheel had been in service beyond its recommended "
        "replacement interval and showed visible surface cracks. Root causes: no replacement date marked on the grinding wheel "
        "and no installation record maintained, pre-use inspection checklist does not include visual wheel surface inspection for "
        "cracks, full face shields stored away from the grinding area making them less convenient than safety glasses. Corrective "
        "actions: introduce grinding wheel installation log recording installation date and replacement interval, update bench "
        "grinder pre-use checklist to include wheel surface visual inspection, relocate full face shields to grinding workstation, "
        "conduct grinding safety refresher for all maintenance staff, inspect and replace all grinding wheels without documented "
        "installation dates."
    ),
    "CAF-2024-001": (
        "Report ID: CAF-2024-001. During aluminium facade installation works at level 22, two rubber packing components used to "
        "secure an aluminium frame on a suspended cradle became unsecured and fell from the cradle platform to the road-side "
        "elevation below. The dropped items landed in a public access area where two vehicles were parked. No damage to vehicles "
        "or injuries to personnel were reported. Root causes: no mandatory pre-use inspection checklist existed for the cradle "
        "platform to verify materials were secured before operations commenced, crew members lacked sufficient awareness of "
        "hazards associated with unsecured materials on cradle platforms operating at height above public access areas. Corrective "
        "actions: implement mandatory pre-use cradle inspection checklist requiring material security verification, require all "
        "materials on cradle platforms to be stored in secured containers during operations, conduct safety awareness briefing on "
        "dropped object hazards for all cradle crews, establish daily safety inspection schedule to assess compliance with cradle "
        "protocols, review and revise method statement and risk assessment for cradle operations."
    ),
    "CAF-2025-002": (
        "Report ID: CAF-2025-002. During a manual material shifting operation involving a team of 20 workers transporting ACP "
        "cladding sheets from the 3rd floor to the ground floor via internal staircase, one operative lost control of a cladding "
        "sheet. The sheet slipped from the operative's grip and struck a colleague on the head causing a laceration. All operatives "
        "were wearing full PPE including helmets and hand gloves. First aid was administered on site and the injured operative was "
        "transported to the nearest clinic. Root causes: task risk assessment did not adequately address the specific hazards of "
        "manually transporting large cladding sheets via internal staircases, no mechanical handling aid or alternative route was "
        "identified despite the size and weight of the panels making manual staircase transport inherently high risk. Corrective "
        "actions: review and revise task risk assessment for manual cladding sheet shifting, assess feasibility of mechanical "
        "handling aids or alternative routes for large panels, conduct manual handling refresher training for cladding installation "
        "operatives, review team size and operative spacing requirements for staircase handling tasks, issue revised safe work "
        "procedure for large panel manual shifting operations."
    ),
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log(msg):
    """Print to console AND append to log file with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def call_ollama(model, prompt, keep_alive=0):
    """Send a prompt to Ollama with explicit context clearing.

    keep_alive=0 means the model is unloaded from memory immediately after
    the request, which guarantees no context carries over to the next call.
    This is the critical fix for the context bleed problem.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def safe_filename(s):
    """Convert any string to a safe filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


# ─── SYSTEM METRICS SAMPLING ──────────────────────────────────────────────────
# Reads /proc directly so we don't depend on vmstat/free being available
# and so we get the same numbers a parallel monitoring session would see.

def _read_cpu_times():
    """Return (idle, total) jiffies from /proc/stat aggregate cpu line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    # parts[0] is 'cpu', parts[1:] are user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
    values = [int(x) for x in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    total = sum(values)
    return idle, total


def _read_mem_kb():
    """Return (used_kb, total_kb) from /proc/meminfo using MemTotal - MemAvailable."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            info[key.strip()] = int(rest.strip().split()[0])
    total = info["MemTotal"]
    available = info.get("MemAvailable", info["MemFree"])
    used = total - available
    return used, total


class SystemMonitor:
    """Background thread that samples CPU% and RAM usage at fixed intervals.

    CPU% is computed across the whole machine (all cores, weighted average),
    matching what `vmstat` reports. To compare with prior `vmstat`-based
    figures the peak value is the most directly comparable metric.

    RAM is reported in GB. The 'baseline' is captured at start_monitoring()
    and the model footprint is computed as peak_used - baseline_used.
    """

    def __init__(self, interval=SAMPLE_INTERVAL):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._cpu_samples = []
        self._ram_used_samples_kb = []
        self._baseline_used_kb = 0
        self._mem_total_kb = 0

    def start(self):
        self._stop.clear()
        self._cpu_samples = []
        self._ram_used_samples_kb = []
        self._baseline_used_kb, self._mem_total_kb = _read_mem_kb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        prev_idle, prev_total = _read_cpu_times()
        while not self._stop.is_set():
            time.sleep(self.interval)
            idle, total = _read_cpu_times()
            d_idle = idle - prev_idle
            d_total = total - prev_total
            cpu_pct = 100.0 * (1 - (d_idle / d_total)) if d_total > 0 else 0.0
            self._cpu_samples.append(cpu_pct)
            prev_idle, prev_total = idle, total

            used_kb, _ = _read_mem_kb()
            self._ram_used_samples_kb.append(used_kb)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self):
        """Return dict of peak/mean CPU% and peak RAM (GB) and model footprint (GB)."""
        peak_cpu = max(self._cpu_samples) if self._cpu_samples else 0.0
        mean_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0
        peak_ram_kb = max(self._ram_used_samples_kb) if self._ram_used_samples_kb else self._baseline_used_kb
        baseline_gb = self._baseline_used_kb / (1024 * 1024)
        peak_ram_gb = peak_ram_kb / (1024 * 1024)
        footprint_gb = (peak_ram_kb - self._baseline_used_kb) / (1024 * 1024)
        total_ram_gb = self._mem_total_kb / (1024 * 1024)
        return {
            "peak_cpu_pct": round(peak_cpu, 1),
            "mean_cpu_pct": round(mean_cpu, 1),
            "baseline_ram_gb": round(baseline_gb, 2),
            "peak_ram_gb": round(peak_ram_gb, 2),
            "model_footprint_gb": round(footprint_gb, 2),
            "total_ram_gb": round(total_ram_gb, 2),
            "samples": len(self._cpu_samples),
        }

# ─── OUTPUT HANDLING ──────────────────────────────────────────────────────────

def setup():
    """Creates output directories and initialise CSV file with headers."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run", "Model", "Report_ID", "Prompt_Type",
            "Tokens_Per_Second", "Total_Time_Seconds",
            "Tokens_Generated", "Prompt_Tokens",
            "Eval_Duration_ns", "Load_Duration_ns",
            "Peak_CPU_Pct", "Mean_CPU_Pct",
            "Baseline_RAM_GB", "Peak_RAM_GB", "Model_Footprint_GB",
            "Sample_Count", "Timestamp",
        ])
    open(LOG_FILE, "w").close()


def append_result(row):
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


def save_response(run_id, model, report_id, prompt_type, response_text):
    fname = f"{run_id:02d}_{safe_filename(model)}_{report_id}_{safe_filename(prompt_type)}.txt"
    path = os.path.join(RESPONSE_DIR, fname)
    with open(path, "w") as f:
        f.write(f"Run: {run_id}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Report: {report_id}\n")
        f.write(f"Prompt type: {prompt_type}\n")
        f.write("=" * 70 + "\n\n")
        f.write(response_text)


# ─── MAIN EXPERIMENT LOOP ─────────────────────────────────────────────────────

def main():
    setup()
    log("=" * 70)
    log("AUTOMATED LLM INFERENCE EXPERIMENT")
    log(f"Models: {MODELS}")
    log(f"Reports: {len(REPORTS)}")
    log(f"Prompt types: {len(PROMPT_TYPES)}")
    log(f"Total runs: {len(MODELS) * len(REPORTS) * len(PROMPT_TYPES)}")
    log(f"System sampling interval: {SAMPLE_INTERVAL}s")
    log(f"Output: {OUTPUT_DIR}")
    log("=" * 70)

    run_id = 0
    total_runs = len(MODELS) * len(REPORTS) * len(PROMPT_TYPES)

    for model in MODELS:
        log(f"\n>>> Starting model: {model}")

        for report_id, report_text in REPORTS.items():
            for prompt_type, prompt_instruction in PROMPT_TYPES.items():
                run_id += 1
                full_prompt = f"{prompt_instruction}\n\nReport: {report_text}"

                log(f"  Run {run_id}/{total_runs}: {model} | {report_id} | {prompt_type}")

                # Start system monitoring BEFORE the inference call so the
                # baseline RAM reading reflects pre-load state and the first
                # CPU sample captures the loading spike.
                monitor = SystemMonitor()
                monitor.start()

                t0 = time.time()
                try:
                    result = call_ollama(model, full_prompt, keep_alive=0)
                except urllib.error.URLError as e:
                    log(f"    ERROR: {e}. Retrying in 30s...")
                    monitor.stop()
                    time.sleep(30)
                    monitor = SystemMonitor()
                    monitor.start()
                    try:
                        result = call_ollama(model, full_prompt, keep_alive=0)
                    except Exception as e2:
                        monitor.stop()
                        log(f"    FAILED on retry: {e2}. Skipping run.")
                        append_result([run_id, model, report_id, prompt_type,
                                       "ERROR", "ERROR", "ERROR", "ERROR",
                                       "ERROR", "ERROR",
                                       "ERROR", "ERROR", "ERROR", "ERROR", "ERROR",
                                       0, datetime.now().isoformat()])
                        continue
                wall = time.time() - t0
                monitor.stop()
                sys_metrics = monitor.summary()

                eval_count    = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 1)
                prompt_count  = result.get("prompt_eval_count", 0)
                total_dur     = result.get("total_duration", 0)
                load_dur      = result.get("load_duration", 0)
                response      = result.get("response", "")

                tps = round(eval_count / (eval_duration / 1e9), 2) if eval_duration > 0 else 0
                total_s = round(total_dur / 1e9, 2)

                log(f"    -> TPS={tps} Time={total_s}s Tokens={eval_count} "
                    f"(wall={wall:.1f}s) CPU_peak={sys_metrics['peak_cpu_pct']}% "
                    f"RAM_peak={sys_metrics['peak_ram_gb']}GB "
                    f"Footprint={sys_metrics['model_footprint_gb']}GB")

                append_result([
                    run_id, model, report_id, prompt_type,
                    tps, total_s, eval_count, prompt_count,
                    eval_duration, load_dur,
                    sys_metrics["peak_cpu_pct"], sys_metrics["mean_cpu_pct"],
                    sys_metrics["baseline_ram_gb"], sys_metrics["peak_ram_gb"],
                    sys_metrics["model_footprint_gb"],
                    sys_metrics["samples"], datetime.now().isoformat(),
                ])
                save_response(run_id, model, report_id, prompt_type, response)

                # Small pause between runs lets the OS release any pending I/O
                # and gives a clean separation between runs in the log.
                # Because keep_alive=0 unloaded the model, this also allows
                # RAM to settle before the next baseline is taken.
                time.sleep(2)

        log(f">>> Completed model: {model}")

    log("\n" + "=" * 70)
    log(f"EXPERIMENT COMPLETE — {run_id} runs")
    log(f"Results CSV: {RESULTS_CSV}")
    log(f"Responses:   {RESPONSE_DIR}/")
    log(f"Log:         {LOG_FILE}")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n!! INTERRUPTED BY USER")
        sys.exit(1)
    except Exception as e:
        log(f"\n!! FATAL ERROR: {e}")
        raise