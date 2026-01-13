import os
import sys
import argparse
import subprocess
import shlex
import re
from typing import Dict, List, Tuple
import os as _os_for_stream  # avoid shadowing

# Ensure project root on sys.path (for consistency with other scripts)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_common_args(args: argparse.Namespace) -> List[str]:
    cmd = []
    if args.window is not None:
        cmd += ["--window", str(args.window)]
    if args.max_kp is not None:
        cmd += ["--max-kp", str(args.max_kp)]
    if args.loops is not None:
        cmd += ["--loops", str(args.loops)]
    if args.yaw_use is not None:
        cmd += ["--yaw-use", args.yaw_use]
    if args.yaw_sign is not None:
        cmd += ["--yaw-sign", str(args.yaw_sign)]
    if args.log_every is not None:
        cmd += ["--log-every", str(args.log_every)]
    if args.no_loop_closure:
        cmd += ["--no-loop-closure"]
    # Pass-through SuperGlue params for parity (ignored by classic scripts)
    if args.sg_match_threshold is not None:
        cmd += ["--sg-match-threshold", str(args.sg_match_threshold)]
    if args.sg_sinkhorn_iters is not None:
        cmd += ["--sg-sinkhorn-iters", str(args.sg_sinkhorn_iters)]
    if args.superpoint_weights is not None:
        cmd += ["--superpoint-weights", args.superpoint_weights]
    if args.superglue_weights is not None:
        cmd += ["--superglue-weights", args.superglue_weights]
    return cmd


def parse_metrics(stdout: str) -> Dict[str, str]:
    """
    Parse the 'Tracking Metrics' block from a script's stdout into a dict.
    """
    metrics: Dict[str, str] = {}
    # Find the last occurrence of the block header to avoid earlier calibrations/logs
    blocks = stdout.split("=== Tracking Metrics ===")
    if len(blocks) < 2:
        return metrics
    tail = blocks[-1]
    # Extract simple key: value pairs by regex
    # Total Frames
    m = re.search(r"Total Frames:\s*([0-9]+)", tail)
    if m:
        metrics["Total Frames"] = m.group(1)
    # Translation Errors
    m = re.search(r"Translation Errors:\s*[\r\n]+(?:\s*)Mean:\s*([0-9.+-eE]+)\s*pixels", tail)
    if m:
        metrics["Trans Mean (px)"] = m.group(1)
    m = re.search(r"Translation Errors:[\s\S]*?Max:\s*([0-9.+-eE]+)\s*pixels", tail)
    if m:
        metrics["Trans Max (px)"] = m.group(1)
    # Rotation Errors
    m = re.search(r"Rotation Errors:\s*[\r\n]+(?:\s*)Mean:\s*([0-9.+-eE]+)\s*degrees", tail)
    if m:
        metrics["Rot Mean (deg)"] = m.group(1)
    m = re.search(r"Rotation Errors:[\s\S]*?Max:\s*([0-9.+-eE]+)\s*degrees", tail)
    if m:
        metrics["Rot Max (deg)"] = m.group(1)
    # Feature Tracking
    m = re.search(r"Mean Features:\s*([0-9.+-eE]+)", tail)
    if m:
        metrics["Mean Features"] = m.group(1)
    m = re.search(r"Mean Inlier Ratio:\s*([0-9.+-eE]+)%", tail)
    if m:
        metrics["Mean Inlier Ratio (%)"] = m.group(1)
    # Trajectory Errors
    m = re.search(r"Absolute Trajectory Error \(ATE\):\s*([0-9.+-eE]+)\s*pixels", tail)
    if m:
        metrics["ATE (px)"] = m.group(1)
    m = re.search(r"Relative Pose Error \(RPE\):\s*([0-9.+-eE]+)\s*pixels", tail)
    if m:
        metrics["RPE (px)"] = m.group(1)
    return metrics


def run_method(
    script_rel: str,
    method_tag: str,
    image: str,
    trajectory: str,
    common_args: List[str],
    extra_args: List[str],
    out_prefix: str,
) -> Tuple[Dict[str, str], str]:
    """
    Run one method script as a subprocess; stream output live and also capture it.
    Returns (metrics_dict, full_stdout_text). Saves two images:
    traj_map_<tag>.png and gt_overlay_<tag>.png in CWD.
    """
    traj_out = f"{out_prefix}traj_map_{method_tag}.png"
    overlay_out = f"{out_prefix}gt_overlay_{method_tag}.png"
    script_path = os.path.join("methods", script_rel)
    cmd = [
        sys.executable, "-u",  # unbuffered for timely streaming
        script_path,
        "--image", image,
        "--trajectory", trajectory,
        "--save-traj-map", traj_out,
        "--save-gt-overlay", overlay_out,
    ] + common_args + extra_args

    # Stream child output live and capture it
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    captured_lines: List[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="")
            captured_lines.append(line)
    proc.wait()
    stdout = "".join(captured_lines)
    metrics = parse_metrics(stdout)
    return metrics, stdout


def main():
    ap = argparse.ArgumentParser(description="Run all VO methods and aggregate metrics.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--max-kp", type=int, default=500)
    ap.add_argument("--yaw-use", choices=["cur", "prev", "mid"], default=None)
    ap.add_argument("--yaw-sign", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--no-loop-closure", action="store_true", help="Forwarded to methods that support it")
    # SuperGlue-related passthroughs
    ap.add_argument("--superpoint-weights", type=str, default=None)
    ap.add_argument("--superglue-weights", type=str, default="outdoor")
    ap.add_argument("--sg-match-threshold", type=float, default=0.2)
    ap.add_argument("--sg-sinkhorn-iters", type=int, default=20)
    # Control which methods to run
    ap.add_argument("--include-vo-only", action="store_true", help="Also run VO-only (no loop closure) variant")
    ap.add_argument("--methods-out-prefix", type=str, default="", help="Optional prefix for output filenames")
    ap.add_argument("--metrics-out", type=str, default="metrics_summary.txt")
    ap.add_argument("--exclude-sg", action="store_true", help="Skip SuperGlue run (classic methods only)")
    args = ap.parse_args()

    common_args = build_common_args(args)
    out_prefix = args.methods_out_prefix or ""

    runs: List[Tuple[str, str, List[str]]] = []
    # SuperGlue with loop-closure (unless excluded)
    if not args.exclude_sg:
        runs.append((
            "superpoint_simulation_aerial_with_loopclosure_posegraph_from_trajectory.py",
            "sg",
            [],  # extra args
        ))
    # Classic methods with loop-closure
    runs.append((
        "superpoint_simulation_aerial_orb_with_loopclosure_posegraph_from_trajectory.py",
        "orb",
        [],
    ))
    runs.append((
        "superpoint_simulation_aerial_sift_with_loopclosure_posegraph_from_trajectory.py",
        "sift",
        [],
    ))
    runs.append((
        "superpoint_simulation_aerial_akaze_with_loopclosure_posegraph_from_trajectory.py",
        "akaze",
        [],
    ))
    # Optional VO-only (no loop closure)
    if args.include_vo_only:
        runs.append((
            "superpoint_simulation_aerial_vo_only_from_trajectory.py",
            "vo",
            [],
        ))

    all_results: List[Tuple[str, Dict[str, str]]] = []
    logs_dir = "method_logs"
    os.makedirs(logs_dir, exist_ok=True)

    for script_rel, tag, extra in runs:
        print(f"[RUN] {tag} -> {script_rel}")
        metrics, stdout = run_method(
            script_rel=script_rel,
            method_tag=tag,
            image=args.image,
            trajectory=args.trajectory,
            common_args=common_args,
            extra_args=extra,
            out_prefix=out_prefix,
        )
        all_results.append((tag, metrics))
        # Save full stdout for debugging
        with open(os.path.join(logs_dir, f"{tag}.log"), "w", encoding="utf-8") as f:
            f.write(stdout)

    # Write combined metrics summary
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        f.write("Method, Total Frames, Trans Mean (px), Trans Max (px), Rot Mean (deg), Rot Max (deg), Mean Features, Mean Inlier Ratio (%), ATE (px), RPE (px)\n")
        for tag, md in all_results:
            row = [
                tag,
                md.get("Total Frames", ""),
                md.get("Trans Mean (px)", ""),
                md.get("Trans Max (px)", ""),
                md.get("Rot Mean (deg)", ""),
                md.get("Rot Max (deg)", ""),
                md.get("Mean Features", ""),
                md.get("Mean Inlier Ratio (%)", ""),
                md.get("ATE (px)", ""),
                md.get("RPE (px)", ""),
            ]
            f.write(", ".join(row) + "\n")
    print("[DONE] Wrote metrics summary ->", args.metrics_out)
    print("[INFO] Individual logs saved in:", logs_dir)


if __name__ == "__main__":
    main()


