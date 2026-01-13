import os
import sys
import math
import json
import argparse
import cv2
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics import TrackingMetrics

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def circ_mean(a0: float, a1: float) -> float:
    return wrap_pi(a0 + 0.5 * wrap_pi(a1 - a0))

def rotate_square(img, angle_rad: float, border_value=0):
    if abs(angle_rad) < 1e-12:
        return img
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

def extract_camera_window(big_bgr, x, y, window_size, yaw_cam_rad, pad_mode="reflect"):
    H, W = big_bgr.shape[:2]
    diag = int(math.ceil(window_size * math.sqrt(2)))
    half = diag // 2
    border = cv2.BORDER_REFLECT101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT
    pad = half + 2
    big_pad = cv2.copyMakeBorder(big_bgr, pad, pad, pad, pad, borderType=border, value=(0, 0, 0))
    xp = int(round(x)) + pad
    yp = int(round(y)) + pad
    patch = big_pad[yp - half: yp + half, xp - half: xp + half].copy()
    patch_rot = rotate_square(patch, yaw_cam_rad, border_value=0)
    ph, pw = patch_rot.shape[:2]
    cx, cy = pw // 2, ph // 2
    ws = window_size // 2
    return patch_rot[cy - ws: cy - ws + window_size, cx - ws: cx - ws + window_size].copy()

def sift_extract(gray: np.ndarray, max_kp: int):
    if not hasattr(cv2, "SIFT_create"):
        return None, None, None
    sift = cv2.SIFT_create(nfeatures=max_kp)
    kpts_cv, desc = sift.detectAndCompute(gray, None)
    if not kpts_cv or desc is None or len(kpts_cv) == 0:
        return None, None, None
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_cv], dtype=np.float32)
    scores = np.array([kp.response for kp in kpts_cv], dtype=np.float32)
    return kpts, desc.astype(np.float32), scores

def sift_match(desc0: np.ndarray, desc1: np.ndarray, ratio=0.75):
    if desc0 is None or desc1 is None:
        return None, None
    index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc0, desc1, k=2)
    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) == 0:
        return None, None
    p0 = np.array([m.queryIdx for m in good], dtype=np.int32)
    p1 = np.array([m.trainIdx for m in good], dtype=np.int32)
    return p0, p1

def estimate_translation_from_pairs(kpts0, kpts1, idx0, idx1, ransac_thresh=3.0):
    if idx0 is None or idx1 is None or len(idx0) < 8:
        return None, 0, 0
    p0 = kpts0[idx0].astype(np.float32)
    p1 = kpts1[idx1].astype(np.float32)
    M, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000, confidence=0.99
    )
    if M is None:
        return None, int(len(idx0)), 0
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    ninl = int(inliers.sum()) if inliers is not None else int(len(idx0))
    return np.array([dx, dy], dtype=np.float32), int(len(idx0)), ninl

def choose_yaw(yaw_prev, yaw_cur, mode: str):
    if mode == "cur":
        return yaw_cur
    if mode == "prev":
        return yaw_prev
    return circ_mean(yaw_prev, yaw_cur)

def calibrate_convention_sift(big, traj_xy_yaw, window, max_kp, yaw_sign,
                              max_test_frames=180, yaw_straight_deg=2.0):
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
    def median_err(cam_sign, yaw_use):
        errs = []
        prev_kpts = None
        prev_desc = None
        for i in range(0, straight_end):
            x, y, yaw_cur = traj_xy_yaw[i]
            yaw_prev = traj_xy_yaw[i-1][2] if i > 0 else yaw_cur
            yaw_used = choose_yaw(yaw_prev, yaw_cur, yaw_use) * yaw_sign
            win = extract_camera_window(big, x, y, window, yaw_cam_rad=cam_sign * yaw_used)
            stab = rotate_square(win, angle_rad=(-cam_sign * yaw_used), border_value=0)
            gray = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            kpts, desc, scores = sift_extract(gray, max_kp)
            if kpts is None:
                prev_kpts = None
                prev_desc = None
                continue
            if prev_kpts is not None and prev_desc is not None:
                i0, i1 = sift_match(prev_desc, desc)
                t, nm, ninl = estimate_translation_from_pairs(prev_kpts, kpts, i0, i1)
                if nm > 0:
                    errs.append(1.0 - (ninl / float(nm)))
            prev_kpts = kpts
            prev_desc = desc
        if len(errs) < 10:
            return 1e9
        return float(np.median(errs))
    best = None
    for cam_sign, yaw_use in candidates:
        e = median_err(cam_sign, yaw_use)
        if best is None or e < best[0]:
            best = (e, cam_sign, yaw_use)
    if best is None:
        return +1, "prev"
    med, cam_best, yaw_best = best
    print("Calibrating rotation convention...")
    print(f"[CALIB] best median error = {med:.3f} (lower is better)")
    print(f"[CALIB] cam_sign={cam_best} yaw_use={yaw_best} yaw_sign={yaw_sign}")
    return cam_best, yaw_best

def should_add_keyframe(est_xy, last_kf_xy, min_dist_px):
    if last_kf_xy is None:
        return True
    return float(np.linalg.norm(est_xy - last_kf_xy)) >= float(min_dist_px)

def apply_tail_correction(est_xy_list, anchor_idx, drift):
    n = len(est_xy_list)
    cur_idx = n - 1
    if anchor_idx < 0 or anchor_idx >= n or cur_idx <= anchor_idx:
        return
    span = cur_idx - anchor_idx
    per = drift / float(span)
    for i in range(anchor_idx + 1, n):
        est_xy_list[i] = est_xy_list[i] + per * float(i - anchor_idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--max-kp", type=int, default=500)
    ap.add_argument("--yaw-use", choices=["cur", "prev", "mid"], default=None)
    ap.add_argument("--yaw-sign", type=float, default=1.0)
    ap.add_argument("--no-loop-closure", action="store_true")
    ap.add_argument("--lc-check-every", type=int, default=25)
    ap.add_argument("--lc-cooldown", type=int, default=200)
    ap.add_argument("--lc-min-sep", type=int, default=400)
    ap.add_argument("--lc-radius", type=float, default=250.0)
    ap.add_argument("--lc-max-candidates", type=int, default=10)
    ap.add_argument("--lc-min-inliers", type=int, default=80)
    ap.add_argument("--kf-min-dist", type=float, default=40.0)
    ap.add_argument("--max-db", type=int, default=2000)
    # keep unused params for CLI parity
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
    sx = sy = 1.0
    if isinstance(meta, dict):
        sz = meta.get("image_size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            mw, mh = float(sz[0]), float(sz[1])
            if mw > 0 and mh > 0:
                sx = W / mw
                sy = H / mh

    print("Device:", "cpu")

    traj_xy_yaw = []
    for p in traj:
        gx = float(p["x"]) * sx
        gy = float(p["y"]) * sy
        gyaw = float(p["yaw"])
        traj_xy_yaw.append((gx, gy, gyaw))

    yaw_sign = float(args.yaw_sign)
    cam_sign, yaw_use_auto = calibrate_convention_sift(
        big, traj_xy_yaw, args.window, args.max_kp, yaw_sign=yaw_sign
    )
    yaw_use = args.yaw_use if args.yaw_use is not None else yaw_use_auto

    metrics = TrackingMetrics()
    x0, y0, yaw0 = traj_xy_yaw[0]
    est_xy = np.array([x0, y0], dtype=np.float32)
    est_xy_list = []
    gt_xy_list = []
    kf_db = []
    last_kf_xy = None
    last_lc_frame = -10**9
    prev_kpts = None
    prev_desc = None

    total_frames = len(traj_xy_yaw) * int(args.loops)
    print("=== Running VO (SIFT) with loop closure ===")
    print(f"poses={len(traj_xy_yaw)}, loops={args.loops}, total_frames={total_frames}")
    print(f"sx={sx:.3f}, sy={sy:.3f}, cam_sign={cam_sign}, yaw_use={yaw_use}, yaw_sign={yaw_sign}")

    frame_idx = 0
    for loop_i in range(args.loops):
        for i, (gx, gy, yaw_cur) in enumerate(traj_xy_yaw):
            yaw_prev = traj_xy_yaw[i-1][2] if i > 0 else yaw_cur
            yaw_used = choose_yaw(yaw_prev, yaw_cur, yaw_use) * yaw_sign
            win = extract_camera_window(big, gx, gy, args.window, yaw_cam_rad=cam_sign * yaw_used)
            stab = rotate_square(win, angle_rad=(-cam_sign * yaw_used), border_value=0)
            gray = cv2.cvtColor(stab, cv2.COLOR_BGR2GRAY)
            kpts, desc, scores = sift_extract(gray, args.max_kp)

            num_matches = 0
            num_inliers = 0
            if kpts is not None and prev_kpts is not None and prev_desc is not None and desc is not None:
                i0, i1 = sift_match(prev_desc, desc)
                t_feat, num_matches, num_inliers = estimate_translation_from_pairs(prev_kpts, kpts, i0, i1)
                if t_feat is not None:
                    est_xy = est_xy + (-t_feat).astype(np.float32)

            gt_xy_list.append((gx, gy))
            est_xy_list.append(est_xy.copy())

            if kpts is not None and desc is not None:
                if should_add_keyframe(est_xy, last_kf_xy, args.kf_min_dist):
                    kf_db.append({
                        "frame": frame_idx,
                        "est_xy": est_xy.copy(),
                        "kpts": kpts,
                        "desc": desc,
                    })
                    last_kf_xy = est_xy.copy()
                    if len(kf_db) > args.max_db:
                        kf_db.pop(0)

            if (not args.no_loop_closure) and (frame_idx % args.lc_check_every == 0):
                if (frame_idx - last_lc_frame) >= args.lc_cooldown and len(kf_db) > 0 and kpts is not None and desc is not None:
                    candidates = []
                    for kf in kf_db:
                        if (frame_idx - kf["frame"]) < args.lc_min_sep:
                            continue
                        if float(np.linalg.norm(est_xy - kf["est_xy"])) > float(args.lc_radius):
                            continue
                        candidates.append(kf)
                    if len(candidates) > args.lc_max_candidates:
                        step = max(1, len(candidates) // args.lc_max_candidates)
                        candidates = candidates[::step][:args.lc_max_candidates]
                    best = None
                    for kf in candidates:
                        i0, i1 = sift_match(kf["desc"], desc)
                        t_kf, nm, ninl = estimate_translation_from_pairs(kf["kpts"], kpts, i0, i1)
                        if nm <= 0:
                            continue
                        if ninl >= args.lc_min_inliers:
                            if best is None or ninl > best[0]:
                                best = (ninl, kf["frame"], kf["est_xy"].copy())
                    if best is not None:
                        ninl, anchor_frame, anchor_est_xy = best
                        cur_est_xy = est_xy_list[-1].copy()
                        drift = (anchor_est_xy - cur_est_xy).astype(np.float32)
                        apply_tail_correction(est_xy_list, anchor_frame, drift)
                        est_xy = est_xy_list[-1].copy()
                        last_lc_frame = frame_idx
                        print(f"[LC] anchor={anchor_frame} -> cur={len(est_xy_list)-1} inliers={ninl} drift=({drift[0]:.1f},{drift[1]:.1f})")

            gt_disp = np.array([gx - (W // 2), gy - (H // 2)], dtype=np.float32)
            est_disp = np.array([est_xy[0] - (W // 2), est_xy[1] - (H // 2)], dtype=np.float32)
            metrics.update_metrics(
                gt_position=gt_disp,
                est_position=est_disp,
                gt_yaw=yaw_cur,
                est_yaw=yaw_used,
                num_features=int(0 if kpts is None else kpts.shape[0]),
                num_inliers=int(num_inliers),
            )

            if args.log_every > 0 and (frame_idx % args.log_every == 0):
                nf = 0 if kpts is None else int(kpts.shape[0])
                print(f"[{loop_i+1}/{args.loops}] frame={frame_idx}/{total_frames} "
                      f"feats={nf} matches={int(num_matches)} inliers={int(num_inliers)} "
                      f"est=({est_xy[0]:.1f},{est_xy[1]:.1f}) yaw_used={math.degrees(yaw_used):.1f}°")

            prev_kpts = kpts
            prev_desc = desc
            frame_idx += 1

    metrics.print_metrics()

    traj_out = args.save_traj_map if args.save_traj_map else "traj_map_sift.png"
    overlay_out = args.save_gt_overlay if args.save_gt_overlay else "gt_overlay_sift.png"

    if traj_out:
        map_size = 900
        canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)
        scale = map_size / float(max(H, W))
        cx, cy = map_size // 2, map_size // 2
        for i in range(1, len(gt_xy_list)):
            x0, y0 = gt_xy_list[i-1]
            x1, y1 = gt_xy_list[i]
            p0 = (int(cx + (x0 - W/2) * scale), int(cy + (y0 - H/2) * scale))
            p1 = (int(cx + (x1 - W/2) * scale), int(cy + (y1 - H/2) * scale))
            cv2.line(canvas, p0, p1, (0, 0, 255), 2)
        for i in range(1, len(est_xy_list)):
            x0, y0 = est_xy_list[i-1]
            x1, y1 = est_xy_list[i]
            p0 = (int(cx + (x0 - W/2) * scale), int(cy + (y0 - H/2) * scale))
            p1 = (int(cx + (x1 - W/2) * scale), int(cy + (y1 - H/2) * scale))
            cv2.line(canvas, p0, p1, (0, 255, 0), 2)
        cv2.imwrite(traj_out, canvas)
        print("[SAVED] traj map ->", traj_out)

    if overlay_out:
        overlay = big.copy()
        for i in range(1, len(gt_xy_list)):
            p0 = (int(gt_xy_list[i-1][0]), int(gt_xy_list[i-1][1]))
            p1 = (int(gt_xy_list[i][0]), int(gt_xy_list[i][1]))
            cv2.line(overlay, p0, p1, (0, 0, 255), 2)
        for i in range(1, len(est_xy_list)):
            p0 = (int(est_xy_list[i-1][0]), int(est_xy_list[i-1][1]))
            p1 = (int(est_xy_list[i][0]), int(est_xy_list[i][1]))
            cv2.line(overlay, p0, p1, (0, 255, 0), 2)
        cv2.imwrite(overlay_out, overlay)
        print("[SAVED] overlay ->", overlay_out)

if __name__ == "__main__":
    main()


