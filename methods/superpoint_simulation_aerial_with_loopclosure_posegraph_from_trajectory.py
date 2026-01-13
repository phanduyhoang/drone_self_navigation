import os
import sys
import math
import json
import argparse
import cv2
import numpy as np
import torch

from metrics import TrackingMetrics
from magicpoint.supereye import SuperPointFrontend

# -----------------------------
# SuperGlue import (SuperGluePretrainedNetwork-master style)
# -----------------------------
_SG_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SuperGluePretrainedNetwork-master"))
if os.path.isdir(_SG_REPO) and _SG_REPO not in sys.path:
    sys.path.insert(0, _SG_REPO)

SuperGlue = None
_import_err = None
for mod_path, cls_name in [
    ("models.superglue", "SuperGlue"),
    ("superglue.models.superglue", "SuperGlue"),
]:
    try:
        mod = __import__(mod_path, fromlist=[cls_name])
        SuperGlue = getattr(mod, cls_name)
        break
    except Exception as e:
        _import_err = e

if SuperGlue is None:
    raise ImportError(
        "Could not import SuperGlue. Make sure SuperGluePretrainedNetwork-master exists.\n"
        f"Last error: {_import_err}"
    )

# -----------------------------
# Math helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def circ_mean(a0: float, a1: float) -> float:
    """Circular midpoint between angles."""
    return wrap_pi(a0 + 0.5 * wrap_pi(a1 - a0))

# -----------------------------
# Rotation / crop helpers
# IMPORTANT: this crop is CENTER-CORRECT even near edges.
# It pads the BIG image first, then crops around the shifted center.
# -----------------------------
def rotate_square(img, angle_rad: float, border_value=0):
    if abs(angle_rad) < 1e-12:
        return img
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad), 1.0)
    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return out

def extract_camera_window(big_bgr, x, y, window_size, yaw_cam_rad, pad_mode="reflect"):
    """
    Simulate camera view:
      1) pad big image so center crop never shifts
      2) crop diag patch centered exactly at (x,y)
      3) rotate by yaw_cam_rad
      4) take centered window_size crop
    """
    H, W = big_bgr.shape[:2]
    diag = int(math.ceil(window_size * math.sqrt(2)))
    half = diag // 2

    if pad_mode == "reflect":
        border = cv2.BORDER_REFLECT101
    else:
        border = cv2.BORDER_CONSTANT

    pad = half + 2
    big_pad = cv2.copyMakeBorder(big_bgr, pad, pad, pad, pad, borderType=border, value=(0, 0, 0))

    xp = int(round(x)) + pad
    yp = int(round(y)) + pad

    patch = big_pad[yp - half: yp + half, xp - half: xp + half].copy()

    # rotate patch to simulate camera yaw
    patch_rot = rotate_square(patch, yaw_cam_rad, border_value=0)

    # center crop to window_size
    ph, pw = patch_rot.shape[:2]
    cx, cy = pw // 2, ph // 2
    ws = window_size // 2
    win = patch_rot[cy - ws: cy - ws + window_size, cx - ws: cx - ws + window_size].copy()
    return win

# -----------------------------
# SuperPoint feature extraction
# -----------------------------
def superpoint_features(sp: SuperPointFrontend, gray_norm: np.ndarray):
    """
    Returns:
      kpts: Nx2 float32
      desc: 256xN float32
      scores: N float32
    """
    pts, desc, _ = sp.run(gray_norm)
    if pts is None or desc is None or pts.shape[1] == 0:
        return None, None, None
    kpts = pts[:2, :].T.astype(np.float32)
    if pts.shape[0] >= 3:
        scores = pts[2, :].astype(np.float32)
    else:
        scores = np.ones((kpts.shape[0],), dtype=np.float32)
    desc = desc.astype(np.float32)  # 256xN
    return kpts, desc, scores

def limit_keypoints(kpts, desc, scores, max_kp):
    n = kpts.shape[0]
    if n <= max_kp:
        return kpts, desc, scores
    idx = np.argsort(scores)[-max_kp:]
    return kpts[idx], desc[:, idx], scores[idx]

# -----------------------------
# SuperGlue inputs / matching
# -----------------------------
def to_torch_feats(kpts, desc, scores, device):
    # keypoints: 1xNx2, descriptors: 1x256xN, scores: 1xN
    k = torch.from_numpy(kpts)[None].to(device)
    d = torch.from_numpy(desc)[None].to(device)
    s = torch.from_numpy(scores)[None].to(device)
    return k, d, s

@torch.no_grad()
def superglue_match(superglue, feats0, feats1, img0, img1):
    k0, d0, s0 = feats0
    k1, d1, s1 = feats1
    data = {
        "keypoints0": k0, "keypoints1": k1,
        "descriptors0": d0, "descriptors1": d1,
        "scores0": s0, "scores1": s1,
        "image0": img0, "image1": img1,
    }
    pred = superglue(data)
    matches0 = pred["matches0"][0].detach().cpu().numpy()          # (N0,)
    mscores0 = pred["matching_scores0"][0].detach().cpu().numpy()  # (N0,)
    return matches0, mscores0

# -----------------------------
# Estimate motion (translation only) from matches
# We work in STABILIZED images, so translation is in WORLD axes already.
# -----------------------------
def estimate_translation_ransac(kpts0, kpts1, matches0, ransac_thresh=3.0):
    idx0 = np.where(matches0 > -1)[0]
    if idx0.size < 8:
        return None, 0, 0

    p0 = kpts0[idx0].astype(np.float32)
    p1 = kpts1[matches0[idx0]].astype(np.float32)

    M, inliers = cv2.estimateAffinePartial2D(
        p0, p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99
    )
    if M is None:
        return None, int(idx0.size), 0

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    ninl = int(inliers.sum()) if inliers is not None else int(idx0.size)
    return np.array([dx, dy], dtype=np.float32), int(idx0.size), ninl

# -----------------------------
# Yaw timing: fix "rotates early" bug
# -----------------------------
def choose_yaw(yaw_prev, yaw_cur, mode: str):
    if mode == "cur":
        return yaw_cur
    if mode == "prev":
        return yaw_prev
    return circ_mean(yaw_prev, yaw_cur)  # "mid"

# -----------------------------
# Auto-calibrate rotation convention (optional but helps A LOT)
# Picks cam_sign and yaw_use that make stabilized consecutive frames have ~zero relative rotation
# We do it on the first straight segment (where yaw changes tiny).
# -----------------------------
def calibrate_convention(big, traj_xy_yaw, window, device, sp, superglue, img_dummy,
                         yaw_sign, max_kp, max_test_frames=180, yaw_straight_deg=2.0):
    # find prefix where yaw change is small (straight-ish)
    yaws = [t[2] for t in traj_xy_yaw]
    straight_end = 0
    for i in range(1, min(len(yaws), max_test_frames)):
        if abs(math.degrees(wrap_pi(yaws[i] - yaws[i-1]))) > yaw_straight_deg:
            break
        straight_end = i
    if straight_end < 20:
        straight_end = min(len(yaws) - 1, 60)

    candidates = []
    for cam_sign in [+1, -1]:
        for yaw_use in ["prev", "mid", "cur"]:
            candidates.append((cam_sign, yaw_use))

    def median_abs_rot_err(cam_sign, yaw_use):
        # we measure how much residual "rotation" is left after stabilization.
        # if convention is correct, stabilized images should have very small relative rotation,
        # which shows up as smaller RANSAC residuals and better inliers.
        errs = []
        prev = None
        prev_feats = None
        prev_kpts = None

        for i in range(0, straight_end):
            x, y, yaw_cur = traj_xy_yaw[i]
            yaw_prev = traj_xy_yaw[i-1][2] if i > 0 else yaw_cur
            yaw_used = choose_yaw(yaw_prev, yaw_cur, yaw_use) * yaw_sign

            # simulate camera (cam_sign decides whether image rotates with yaw or opposite)
            win = extract_camera_window(big, x, y, window, yaw_cam_rad=cam_sign * yaw_used)

            # stabilize to north-up: rotate back by (-cam_sign*yaw_used)
            stab = rotate_square(win, angle_rad=(-cam_sign * yaw_used), border_value=0)

            gray = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            gray_norm = gray.astype(np.float32) / 255.0

            kpts, desc, scores = superpoint_features(sp, gray_norm)
            if kpts is None:
                prev = stab
                prev_feats = None
                prev_kpts = None
                continue
            kpts, desc, scores = limit_keypoints(kpts, desc, scores, max_kp)
            feats = to_torch_feats(kpts, desc, scores, device)

            if prev_feats is not None:
                matches0, _ = superglue_match(superglue, prev_feats, feats, img_dummy, img_dummy)
                t, nm, ninl = estimate_translation_ransac(prev_kpts, kpts, matches0)
                # use inverse of inlier ratio as "error" proxy
                if nm > 0:
                    err = 1.0 - (ninl / float(nm))
                    errs.append(err)

            prev_feats = feats
            prev_kpts = kpts

        if len(errs) < 10:
            return 1e9
        return float(np.median(errs))

    best = None
    for cam_sign, yaw_use in candidates:
        e = median_abs_rot_err(cam_sign, yaw_use)
        if best is None or e < best[0]:
            best = (e, cam_sign, yaw_use)

    if best is None:
        return +1, "prev"

    med_err, cam_sign_best, yaw_use_best = best
    print("Calibrating rotation convention...")
    print(f"[CALIB] best median error = {med_err:.3f} (lower is better)")
    print(f"[CALIB] cam_sign={cam_sign_best} yaw_use={yaw_use_best} yaw_sign={yaw_sign}")
    return cam_sign_best, yaw_use_best

# -----------------------------
# Loop closure helpers (fast)
# -----------------------------
def should_add_keyframe(est_xy, last_kf_xy, min_dist_px):
    if last_kf_xy is None:
        return True
    return float(np.linalg.norm(est_xy - last_kf_xy)) >= float(min_dist_px)

def apply_tail_correction(est_xy_list, anchor_idx, drift):
    """Distribute drift linearly from anchor_idx..end."""
    n = len(est_xy_list)
    cur_idx = n - 1
    if anchor_idx < 0 or anchor_idx >= n or cur_idx <= anchor_idx:
        return
    span = cur_idx - anchor_idx
    per = drift / float(span)
    for i in range(anchor_idx + 1, n):
        est_xy_list[i] = est_xy_list[i] + per * float(i - anchor_idx)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--loops", type=int, default=1)

    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--max-kp", type=int, default=500)

    # yaw fix controls
    ap.add_argument("--yaw-use", choices=["cur", "prev", "mid"], default=None,
                    help="Yaw timing for stabilization. If omitted, auto-calibration decides.")
    ap.add_argument("--yaw-sign", type=float, default=1.0,
                    help="Flip yaw direction (+1 or -1). Leave +1 unless convention mismatch.")

    ap.add_argument("--no-loop-closure", action="store_true")

    # loop closure speed + behavior
    ap.add_argument("--lc-check-every", type=int, default=25)
    ap.add_argument("--lc-cooldown", type=int, default=200)
    ap.add_argument("--lc-min-sep", type=int, default=400)
    ap.add_argument("--lc-radius", type=float, default=250.0, help="candidate filter radius in estimated coords")
    ap.add_argument("--lc-max-candidates", type=int, default=10)
    ap.add_argument("--lc-min-inliers", type=int, default=80)

    ap.add_argument("--kf-min-dist", type=float, default=40.0)
    ap.add_argument("--max-db", type=int, default=2000)

    # models
    ap.add_argument("--superpoint-weights", type=str, default=None)
    ap.add_argument("--superglue-weights", type=str, default="outdoor")
    ap.add_argument("--sg-match-threshold", type=float, default=0.2)
    ap.add_argument("--sg-sinkhorn-iters", type=int, default=20)

    ap.add_argument("--log-every", type=int, default=200)

    ap.add_argument("--save-traj-map", type=str, default=None)
    ap.add_argument("--save-gt-overlay", type=str, default=None)

    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    if not os.path.exists(args.trajectory):
        raise FileNotFoundError(args.trajectory)

    big = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if big is None:
        raise RuntimeError("Could not load image")

    H, W = big.shape[:2]

    with open(args.trajectory, "r") as f:
        td = json.load(f)
    traj = td.get("trajectory", [])
    meta = td.get("meta", {})

    if not traj:
        raise RuntimeError("Trajectory JSON contains no 'trajectory' list")

    # scale trajectory coordinates if meta.image_size exists
    sx = sy = 1.0
    if isinstance(meta, dict):
        sz = meta.get("image_size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            mw, mh = float(sz[0]), float(sz[1])
            if mw > 0 and mh > 0:
                sx = W / mw
                sy = H / mh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # SuperPoint (GPU)
    if args.superpoint_weights is None:
        args.superpoint_weights = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "magicpoint", "superpoint_v1.pth")
        )
    sp = SuperPointFrontend(
        args.superpoint_weights,
        nms_dist=4,
        conf_thresh=0.015,
        nn_thresh=0.7,
        cuda=(device.type == "cuda")
    )

    # SuperGlue (GPU)
    sg_conf = {
        "weights": args.superglue_weights,
        "sinkhorn_iterations": args.sg_sinkhorn_iters,
        "match_threshold": args.sg_match_threshold,
    }
    superglue = SuperGlue(sg_conf).to(device).eval()
    print(f"Loaded SuperGlue model ({args.superglue_weights} weights)")

    # dummy image tensor for SuperGlue normalization
    img_dummy = torch.zeros((1, 1, args.window, args.window), dtype=torch.float32, device=device)

    # Build scaled trajectory list
    traj_xy_yaw = []
    for p in traj:
        gx = float(p["x"]) * sx
        gy = float(p["y"]) * sy
        gyaw = float(p["yaw"])
        traj_xy_yaw.append((gx, gy, gyaw))

    # auto-calibrate cam_sign + yaw_use unless user forces yaw_use
    yaw_sign = float(args.yaw_sign)
    cam_sign, yaw_use_auto = calibrate_convention(
        big, traj_xy_yaw, args.window, device, sp, superglue, img_dummy,
        yaw_sign=yaw_sign, max_kp=args.max_kp
    )
    yaw_use = args.yaw_use if args.yaw_use is not None else yaw_use_auto

    # Metrics
    metrics = TrackingMetrics()

    # Estimate state: keep EST in IMAGE PIXELS (not displacement), so overlay is direct.
    x0, y0, yaw0 = traj_xy_yaw[0]
    est_xy = np.array([x0, y0], dtype=np.float32)
    est_xy_list = []
    gt_xy_list = []

    # Keyframe DB stores GPU-ready features to avoid re-uploading every time
    # entry: { 'frame': int, 'est_xy': np.array(2), 'kpts': np, 'feats': (k,d,s) torch }
    kf_db = []
    last_kf_xy = None
    last_lc_frame = -10**9

    prev_kpts = None
    prev_feats = None

    total_frames = len(traj_xy_yaw) * int(args.loops)
    print("=== Running VO (SuperPoint+SuperGlue) with ORACLE YAW stabilization ===")
    print(f"poses={len(traj_xy_yaw)}, loops={args.loops}, total_frames={total_frames}")
    print(f"sx={sx:.3f}, sy={sy:.3f}, cam_sign={cam_sign}, yaw_use={yaw_use}, yaw_sign={yaw_sign}")

    frame_idx = 0

    for loop_i in range(args.loops):
        for i, (gx, gy, yaw_cur) in enumerate(traj_xy_yaw):
            yaw_prev = traj_xy_yaw[i-1][2] if i > 0 else yaw_cur
            yaw_used = choose_yaw(yaw_prev, yaw_cur, yaw_use) * yaw_sign

            # simulate camera view + stabilize to north-up
            win = extract_camera_window(big, gx, gy, args.window, yaw_cam_rad=cam_sign * yaw_used)
            stab = rotate_square(win, angle_rad=(-cam_sign * yaw_used), border_value=0)

            gray = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            gray_norm = gray.astype(np.float32) / 255.0

            kpts, desc, scores = superpoint_features(sp, gray_norm)
            if kpts is None:
                # still log GT/EST
                gt_xy_list.append((gx, gy))
                est_xy_list.append(est_xy.copy())
                frame_idx += 1
                continue

            kpts, desc, scores = limit_keypoints(kpts, desc, scores, args.max_kp)
            feats = to_torch_feats(kpts, desc, scores, device)

            # VO step: match prev->cur in STABILIZED images => translation is in world axes
            num_matches = 0
            num_inliers = 0

            if prev_feats is not None and prev_kpts is not None:
                matches0, _ = superglue_match(superglue, prev_feats, feats, img_dummy, img_dummy)
                t_feat, num_matches, num_inliers = estimate_translation_ransac(prev_kpts, kpts, matches0)
                if t_feat is not None:
                    # feature motion old->new, camera motion is opposite
                    cam_t = (-t_feat).astype(np.float32)
                    est_xy = est_xy + cam_t

            # store trajectories
            gt_xy_list.append((gx, gy))
            est_xy_list.append(est_xy.copy())

            # Keyframe insertion (estimated-based, no GT leakage)
            if should_add_keyframe(est_xy, last_kf_xy, args.kf_min_dist):
                kf_db.append({
                    "frame": frame_idx,
                    "est_xy": est_xy.copy(),
                    "kpts": kpts,
                    "feats": feats,  # keep on GPU
                })
                last_kf_xy = est_xy.copy()
                if len(kf_db) > args.max_db:
                    kf_db.pop(0)

            # Loop closure (optional)
            if (not args.no_loop_closure) and (frame_idx % args.lc_check_every == 0):
                if (frame_idx - last_lc_frame) >= args.lc_cooldown and len(kf_db) > 0:
                    # candidate filter: far in time + near in estimated position
                    candidates = []
                    for kf in kf_db:
                        if (frame_idx - kf["frame"]) < args.lc_min_sep:
                            continue
                        if float(np.linalg.norm(est_xy - kf["est_xy"])) > float(args.lc_radius):
                            continue
                        candidates.append(kf)

                    # limit candidates
                    if len(candidates) > args.lc_max_candidates:
                        step = max(1, len(candidates) // args.lc_max_candidates)
                        candidates = candidates[::step][:args.lc_max_candidates]

                    best = None  # (inliers, anchor_frame, anchor_est_xy)
                    for kf in candidates:
                        m0, _ = superglue_match(superglue, kf["feats"], feats, img_dummy, img_dummy)
                        # estimate translation between kf and current (both stabilized)
                        t_kf, nm, ninl = estimate_translation_ransac(kf["kpts"], kpts, m0)
                        if nm <= 0:
                            continue
                        if ninl >= args.lc_min_inliers:
                            if best is None or ninl > best[0]:
                                best = (ninl, kf["frame"], kf["est_xy"].copy())

                    if best is not None:
                        ninl, anchor_frame, anchor_est_xy = best
                        cur_est_xy = est_xy_list[-1].copy()
                        drift = (anchor_est_xy - cur_est_xy).astype(np.float32)

                        # apply correction to tail (anchor->current)
                        apply_tail_correction(est_xy_list, anchor_frame, drift)
                        est_xy = est_xy_list[-1].copy()
                        last_lc_frame = frame_idx
                        print(f"[LC] anchor={anchor_frame} -> cur={len(est_xy_list)-1} inliers={ninl} drift=({drift[0]:.1f},{drift[1]:.1f})")

            # Metrics (as displacement-from-center)
            gt_disp = np.array([gx - (W // 2), gy - (H // 2)], dtype=np.float32)
            est_disp = np.array([est_xy[0] - (W // 2), est_xy[1] - (H // 2)], dtype=np.float32)
            metrics.update_metrics(
                gt_position=gt_disp,
                est_position=est_disp,
                gt_yaw=yaw_cur,
                est_yaw=yaw_used,
                num_features=int(kpts.shape[0]),
                num_inliers=int(num_inliers),
            )

            if args.log_every > 0 and (frame_idx % args.log_every == 0):
                print(f"[{loop_i+1}/{args.loops}] frame={frame_idx}/{total_frames} "
                      f"feats={int(kpts.shape[0])} matches={int(num_matches)} inliers={int(num_inliers)} "
                      f"est=({est_xy[0]:.1f},{est_xy[1]:.1f}) yaw_used={math.degrees(yaw_used):.1f}°")

            prev_kpts = kpts
            prev_feats = feats
            frame_idx += 1

    # Print metrics
    metrics.print_metrics()

    # Save map (simple dark canvas with GT red + EST green)
    if args.save_traj_map:
        map_size = 900
        canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)
        scale = map_size / float(max(H, W))
        cx, cy = map_size // 2, map_size // 2

        # draw GT
        for i in range(1, len(gt_xy_list)):
            x0, y0 = gt_xy_list[i-1]
            x1, y1 = gt_xy_list[i]
            p0 = (int(cx + (x0 - W/2) * scale), int(cy + (y0 - H/2) * scale))
            p1 = (int(cx + (x1 - W/2) * scale), int(cy + (y1 - H/2) * scale))
            cv2.line(canvas, p0, p1, (0, 0, 255), 2)

        # draw EST
        for i in range(1, len(est_xy_list)):
            x0, y0 = est_xy_list[i-1]
            x1, y1 = est_xy_list[i]
            p0 = (int(cx + (x0 - W/2) * scale), int(cy + (y0 - H/2) * scale))
            p1 = (int(cx + (x1 - W/2) * scale), int(cy + (y1 - H/2) * scale))
            cv2.line(canvas, p0, p1, (0, 255, 0), 2)

        cv2.imwrite(args.save_traj_map, canvas)
        print("[SAVED] traj map ->", args.save_traj_map)

    # Save overlay on big image (GT red + EST green)
    if args.save_gt_overlay:
        overlay = big.copy()

        for i in range(1, len(gt_xy_list)):
            p0 = (int(gt_xy_list[i-1][0]), int(gt_xy_list[i-1][1]))
            p1 = (int(gt_xy_list[i][0]), int(gt_xy_list[i][1]))
            cv2.line(overlay, p0, p1, (0, 0, 255), 2)

        for i in range(1, len(est_xy_list)):
            p0 = (int(est_xy_list[i-1][0]), int(est_xy_list[i-1][1]))
            p1 = (int(est_xy_list[i][0]), int(est_xy_list[i][1]))
            cv2.line(overlay, p0, p1, (0, 255, 0), 2)

        cv2.imwrite(args.save_gt_overlay, overlay)
        print("[SAVED] overlay ->", args.save_gt_overlay)


if __name__ == "__main__":
    main()


