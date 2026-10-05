import pathlib
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_numeric(val_str: str) -> float:
    """Converts metric string representations like '28.2K' or '1.5M' to floats."""
    val_str = val_str.strip().upper()
    multiplier = 1.0
    if val_str.endswith("K"):
        multiplier = 1_000.0
        val_str = val_str[:-1]
    elif val_str.endswith("M"):
        multiplier = 1_000_000.0
        val_str = val_str[:-1]
    return float(val_str) * multiplier


def parse_log(file_path: str):
    """
    Parses the log file into separate iterations.
    Detects a new iteration when:
      1. A non-throughput boundary line appears (e.g. 'Using GPU...').
      2. Elapsed time decreases/resets.
    """
    pattern = re.compile(
        r"\[Throughput\]\s*([\d\.]+)s\s*::\s*(\d+)/\d+\s*games\s*::.*?SENT:\s*([\d\.]+[KMkm]?)\s*nps.*?REC:\s*([\d\.]+[KMkm]?)\s*nps"
    )

    iterations = []
    current_run: dict = {"time": [], "games": [], "sent_nps": [], "rec_nps": []}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect marker for a new iteration run
            if "Using GPU" in line:
                if current_run["time"]:
                    iterations.append(current_run)
                    current_run = {"time": [], "games": [], "sent_nps": [], "rec_nps": []}
                continue

            match = pattern.search(line)
            if match:
                time_elapsed = float(match.group(1))
                games = int(match.group(2))
                sent_nps = parse_numeric(match.group(3))
                rec_nps = parse_numeric(match.group(4))

                cum_time = time_elapsed + current_run["time"][-1] if current_run["time"] else time_elapsed

                current_run["time"].append(cum_time)
                current_run["games"].append(games)
                current_run["sent_nps"].append(sent_nps)
                current_run["rec_nps"].append(rec_nps)

    if current_run["time"]:
        iterations.append(current_run)

    return iterations

def compute_nodes_remaining(times, nps_values):
    """
    Integrates SENT nps over time using the trapezoidal rule:
    Total nodes = sum(nps * dt).
    Nodes remaining = Total nodes - cumulative nodes evaluated.
    """
    times = np.array(times)
    nps_values = np.array(nps_values)

    if len(times) <= 1:
        return np.array([0.0])

    dt = np.diff(times)
    # Average nps across interval multiplied by delta t
    interval_nodes = 0.5 * (nps_values[:-1] + nps_values[1:]) * dt
    cumulative_nodes = np.insert(np.cumsum(interval_nodes), 0, 0.0)

    total_nodes = cumulative_nodes[-1]
    nodes_remaining = total_nodes - cumulative_nodes
    return nodes_remaining

def compute_sma(data, window=5):
    """
    Computes a causal (backward-looking) Simple Moving Average.
    Early elements where count < window are averaged up to their index.
    """
    data_arr = np.array(data)
    sma = np.zeros_like(data_arr, dtype=float)
    for i in range(len(data_arr)):
        start_idx = max(0, i - window + 1)
        sma[i] = np.mean(data_arr[start_idx:i+1])
    return sma

def plot_iterations(iterations, save_folder: pathlib.Path):
    """Plots each iteration separately on its own figure with dual y-axes."""
    for idx, data in enumerate(iterations, start=1):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        times = data["time"]
        sent_nps = data["sent_nps"]
        rec_nps = data["rec_nps"]
        games = data["games"]
        nodes_rem = compute_nodes_remaining(times, rec_nps)

        SMA = 20
        sent_nps_sma = compute_sma(sent_nps, window=SMA)
        rec_nps_sma = compute_sma(rec_nps, window=SMA)

        # Primary Y-Axis: SENT & REC nps
        ax1.set_xlabel("Time (s)", fontweight="bold")
        ax1.set_ylabel("SENT / REC nps", color="black", fontweight="bold")
        ax1.set_ylim(0, 40_000)
        
# 1. Plot raw markers (no lines, slightly faded)
        line1_sent_raw = ax1.plot(
            times,
            sent_nps,
            color="tab:blue",
            marker="o",
            markersize=1,
            linestyle="",     # Removes the line between dots
            alpha=0.4,        # Makes markers slightly transparent so the SMA line pops
            label="SENT nps (raw)",
        )
        line1_rec_raw = ax1.plot(
            times,
            rec_nps,
            color="tab:red",
            marker="v",
            markersize=1,
            linestyle="",     # Removes the line between dots
            alpha=0.4,
            label="REC nps (raw)",
        )

        # 2. Plot SMA lines (no markers)
        line1_sent_sma = ax1.plot(
            times,
            sent_nps_sma,
            color="tab:blue",
            linewidth=1,
            linestyle="-",    # Explicitly solid line
            label=f"SENT nps ({SMA}-SMA)",
        )
        line1_rec_sma = ax1.plot(
            times,
            rec_nps_sma,
            color="tab:red",
            linewidth=1,
            linestyle="-",    # Explicitly solid line
            label=f"REC nps ({SMA}-SMA)",
        )

        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Secondary Y-Axis: Completed Games
        ax2 = ax1.twinx()
        color_games = "tab:orange"
        ax2.set_ylabel("Completed Games", color=color_games, fontweight="bold")
        line2 = ax2.plot(
            times,
            games,
            color=color_games,
            marker="s",
            markersize=1,
            linestyle="--",
            label="Games",
        )
        ax2.tick_params(axis="y", labelcolor=color_games)

        # Axis 3: Nodes Remaining
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        color_nodes = "tab:green"
        ax3.set_ylabel(
            "Nodes Remaining in Iteration", color=color_nodes, fontweight="bold"
        )
        line3 = ax3.plot(
            times,
            nodes_rem,
            color=color_nodes,
            marker="^",
            markersize=1,
            linestyle="-.",
            label="Nodes Remaining",
        )
        ax3.tick_params(axis="y", labelcolor=color_nodes)
        ax3.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, _: (
                    f"{y / 1e6:.1f}M"
                    if y >= 1e6
                    else f"{int(y / 1e3)}K"
                    if y >= 1e3
                    else f"{int(y)}"
                )
            )
        )

        # Combined legend
        lines = line1_sent_sma + line1_rec_sma + line1_sent_raw + line1_rec_raw + line2 + line3
        labels: list[str] = [str(l.get_label()) for l in lines]

        ax1.legend(handles=lines, labels=labels, loc="upper right")

        plt.title(f"Iteration {idx}: SENT NPS & Games over Time", fontsize=12)
        fig.tight_layout()
        
        plt.savefig(save_folder / str(idx))


if __name__ == "__main__":
    # Replace 'throughput.log' with your actual log file path
    log_file = sys.argv[1]
    save_folder = pathlib.Path(sys.argv[2])
    assert save_folder.is_dir()

    iterations_data = parse_log(log_file)
    print(f"Loaded {len(iterations_data)} iteration(s). Generating plots...")
    plot_iterations(iterations_data, save_folder)
