Local LLMs in Manufacturing SMEs: Experiment Code

This repository contains the automated experiment script for evaluating local LLMs.
The script is designed to process manufacturing safety reports, as described in the bachelor’s thesis: Local LLMs in Manufacturing SMEs: Feasibility, Privacy, and Cybersecurity Assessment (Karelia University of Applied Sciences, 2026).

The script runs 80 inference combinations: two models, ten safety reports, and four prompt types. It uses the Ollama HTTP API to capture inference metrics from the
API response and system metrics (CPU and RAM) from a background sampling thread.
The script is fully self-contained. The safety reports are included as string constants
and the only runtime requirements are Python 3 (standard library only) and a
running Ollama installation with the required models downloaded.

What this script does

For every run it:

1. Constructs a single-turn prompt by concatenating a task instruction with the
full text of a safety report.
2. It submits the prompt to Ollama at http://localhost:11434/api/generate with
keep_alive: 0, so the model is unloaded from memory after each response.
This explicit context clearing prevents cross-run context bleed that
was seen in earlier interactive testing.
3. Reads inference metrics directly from the API response: eval_count,
eval_duration, prompt_eval_count, total_duration, load_duration.
4. It samples system metrics in a parallel thread by reading /proc/stat and
/proc/meminfo every second during the inference call,
recording peak and mean CPU utilization and peak RAM usage.
5. It writes one CSV row for each run, along with a response text file for each run.

Requirements

* Python 3.8 or newer (standard library only; no pip install needed)
* Ollama installed and running on localhost:11434
* The following models pulled via ollama pull:
  * llama3.1:latest
  * qwen2.5:32b
* Linux host (the system monitor reads from /proc, but the rest of the script
is platform-independent)

How to run

# 1. Verify Ollama is running and models are available
curl http://localhost:11434/api/tags
ollama list

# 2. Run the experiment (takes ~90 minutes on the reference hardware)
python3 run_experiment.py

Output is written to /tmp/experiment_results/:

/tmp/experiment_results/
├── results.csv          # all 80 runs, all metrics
├── responses/           # full model response text per run (80 files)
└── experiment.log       # timestamped run log

To test only a subset, temporarily edit the MODELS or REPORTS
constants near the top of the script.

Reference hardware

The thesis results were produced on a CPU-only virtual machine at Karelia
University of Applied Sciences:

Column 1	Column 2
Component	Specification
CPU	Intel Xeon SapphireRapids, 32 vCPUs (4 sockets × 8)
RAM	64 GB
Storage	1.5 TB
OS	Ubuntu 24.04.2 LTS, Linux 6.8.0-106-generic
Hypervisor	KVM full virtualization
Ollama	0.13.0
Quantization	Q4_K_M for both models


Throughput depends on the hardware, so results on different CPU configurations
will vary.

Reproducing the thesis results

Running the script as-is on a comparable Linux host with the two models pulled
should produce a results.csv whose aggregate values closely match those
reported in Chapter 4 of the thesis. The variance for a single run is small (TPS SD
usually under 1.0 for LLaMA and under 0.1 for Qwen on the reference
hardware), so individual run numbers may not match exactly, but per-prompt
means and standard deviations should fall within the ranges reported.

The qualitative output scores in the thesis were assigned by a
single researcher using the rubric defined in section 3.3.1. Independent
re-scoring of model output is advised.

Citation

If you use this script or the methodology in your own work, please cite the
underlying thesis:

Mohammed, I. (2026). Local LLMs in Manufacturing SMEs: Feasibility, Privacy,
and Cybersecurity Assessment. Bachelor’s thesis. Karelia University of
Applied Sciences, Joensuu, Finland.

License

MIT License — see https://en.wikipedia.org/wiki/MIT_License.

