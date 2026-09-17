"""
Scale-Aware Coarse-to-Fine Matcher for Extreme Pixel Scale Disparity.

Designed specifically for ISRO mission data where:
 - Source image (e.g. OHRC) is ~1200x8000 at 0.25m GSD  (higher resolution)
 - Reference orthostrip (e.g. TMC-2) is ~2532x52224 at 5m GSD (lower resolution)
 - Pixel scale ratio ≈ 6-7x

Correct three-phase strategy
─────────────────────────────
Phase 1 — GSD Normalisation
    Scale the source DOWN by the pixel_scale_ratio so it is expressed in the
    same pixel units as the reference (~184×1227 for the typical OHRC/TMC-2 pair).

Phase 2 — Coarse NCC Localisation
    Build a modest pyramid of the reference and slide the GSD-normalised source
    template across it to find the best candidate overlap row-band.

Phase 3 — Fine SIFT Matching at Common Scale
    Crop the reference to the localised ROI (+ generous margin), resize both
    GSD-normalised images to SIFT_MAX_DIM, then run SIFT + FLANN.
    Map keypoints back to NATIVE pixel coordinates of their original images.
"""
import time
import numpy as np
import cv2
from utils.logging import get_logger

logger = get_logger("ScaleAwareMatcher")

# ── tuneable constants ────────────────────────────────────────────────────────
# Maximum height (in reference pixel units) for the coarse NCC pyramid search.
# Large enough to keep the template at ≥25px for reliable NCC.
_NCC_MAX_DIM   = 8192
# Maximum dimension for fine SIFT stage.  ≥2000 keeps SIFT from missing features.
_SIFT_MAX_DIM  = 3000
# Minimum template size in the coarse frame before NCC is deemed unreliable.
_MIN_TMPL_SIZE = 12

# ── helpers ──────────────────────────────────────────────────────────────────

def _resize_keep_ar(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize so max(h, w) == max_dim; return (resized, scale_factor)."""
    h, w = img.shape[:2]
    mx = max(h, w)
    if mx <= max_dim:
        return img.copy(), 1.0
    sf = max_dim / mx
    new_w = max(1, int(round(w * sf)))
    new_h = max(1, int(round(h * sf)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, sf


def _extract_sift(
    img: np.ndarray,
    n: int = 5000,
    contrast: float = 0.03,
) -> tuple[np.ndarray, np.ndarray | None]:
    """SIFT keypoints + 128-D descriptors from uint8 single-channel image."""
    sift = cv2.SIFT_create(nfeatures=n, contrastThreshold=contrast, edgeThreshold=10.0)
    kps, descs = sift.detectAndCompute(img, None)
    if not kps or descs is None:
        return np.empty((0, 2), dtype=np.float32), None
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return pts, descs.astype(np.float32)


def _match_flann(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FLANN kNN + Lowe's ratio test → (idx1, idx2, confidences)."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        empty = np.empty((0,), dtype=int)
        return empty, empty, np.empty((0,), dtype=np.float32)

    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc1, desc2, k=2)

    idx1, idx2, confs = [], [], []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                confs.append(float(np.clip(1.0 - m.distance / (n.distance + 1e-7), 0, 1)))

    return (
        np.array(idx1, dtype=int),
        np.array(idx2, dtype=int),
        np.array(confs, dtype=np.float32),
    )


def _ncc_localise(
    src_at_ref_gsd: np.ndarray,
    reference_u8: np.ndarray,
    ncc_max_dim: int = _NCC_MAX_DIM,
) -> tuple[int, int, int, int, float]:
    """
    Search for the GSD-normalised source template inside the reference strip.

    Both images must already be expressed in reference pixel units.
    Works at a common coarse scale to keep memory manageable.

    Returns (rx1, ry1, rx2, ry2, ncc_confidence) in NATIVE reference coordinates.
    """
    h_s, w_s = src_at_ref_gsd.shape[:2]
    h_r, w_r = reference_u8.shape[:2]

    # Scale factor that brings the REFERENCE down to ncc_max_dim
    sf = min(1.0, ncc_max_dim / max(h_r, w_r))

    ref_coarse = cv2.resize(reference_u8,
                            (max(1, int(w_r * sf)), max(1, int(h_r * sf))),
                            interpolation=cv2.INTER_AREA).astype(np.float32)
    tmpl_coarse = cv2.resize(src_at_ref_gsd,
                             (max(1, int(w_s * sf)), max(1, int(h_s * sf))),
                             interpolation=cv2.INTER_AREA).astype(np.float32)

    h_tc, w_tc = tmpl_coarse.shape[:2]
    h_rc, w_rc = ref_coarse.shape[:2]

    logger.info(
        f"NCC Localise: coarse_ref={w_rc}x{h_rc}, coarse_tmpl={w_tc}x{h_tc} (sf={sf:.4f})"
    )

    if h_tc < _MIN_TMPL_SIZE or w_tc < _MIN_TMPL_SIZE:
        logger.info(
            f"NCC Localise: template too small ({w_tc}x{h_tc}px) — skipping NCC, using full reference."
        )
        return 0, 0, w_r, h_r, 0.0

    if h_rc < h_tc or w_rc < w_tc:
        return 0, 0, w_r, h_r, 0.0

    # Multi-scale NCC within the coarse space
    best_val = -2.0
    best_loc_c = (0, 0)

    for lvl_sf in [1.0, 0.5, 0.25]:
        t = cv2.resize(tmpl_coarse,
                       (max(4, int(w_tc * lvl_sf)), max(4, int(h_tc * lvl_sf))),
                       interpolation=cv2.INTER_AREA)
        r = cv2.resize(ref_coarse,
                       (max(4, int(w_rc * lvl_sf)), max(4, int(h_rc * lvl_sf))),
                       interpolation=cv2.INTER_AREA)
        if r.shape[0] < t.shape[0] or r.shape[1] < t.shape[1]:
            continue
        try:
            result = cv2.matchTemplate(r, t, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(result)
        except Exception:
            continue

        if val > best_val:
            best_val = float(val)
            # Map loc back to coarse (lvl_sf=1.0) coordinates
            best_loc_c = (int(loc[0] / lvl_sf), int(loc[1] / lvl_sf))

    # Build ROI box in coarse coordinates (2× template size around match)
    bx, by = best_loc_c
    rx1_c = max(0, bx - w_tc)
    ry1_c = max(0, by - h_tc)
    rx2_c = min(w_rc, bx + w_tc * 2)
    ry2_c = min(h_rc, by + h_tc * 2)

    # Map ROI back to native reference coordinates
    rx1_n = int(rx1_c / sf)
    ry1_n = int(ry1_c / sf)
    rx2_n = int(rx2_c / sf)
    ry2_n = int(ry2_c / sf)

    logger.info(
        f"NCC Localise: best match at coarse ({bx},{by}), "
        f"NCC={best_val:.3f}, ROI_native=[{rx1_n},{ry1_n},{rx2_n},{ry2_n}]"
    )
    return rx1_n, ry1_n, rx2_n, ry2_n, best_val


# ── main matcher ─────────────────────────────────────────────────────────────

class ScaleAwareMatcher:
    """
    Coarse-to-fine matcher that handles extreme pixel scale ratios (up to ~20×).

    All keypoints are returned in NATIVE pixel coordinates of their respective images
    (source and reference), ready for RANSAC geometric estimation.
    """

    def find_matches(
        self,
        source_u8: np.ndarray,
        reference_u8: np.ndarray,
        ransac_threshold: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """
        Returns
        -------
        src_pts_native  : (N, 2) float32 — source image pixel coordinates
        ref_pts_native  : (N, 2) float32 — reference image pixel coordinates
        confidences     : (N,) float32
        runtime_s       : float
        diagnostics     : dict
        """
        t0   = time.perf_counter()
        diag: dict = {}

        h_s, w_s = source_u8.shape[:2]
        h_r, w_r = reference_u8.shape[:2]

        # Pixel scale ratio: how many reference pixels per source pixel (>1 → ref coarser)
        pixel_scale_ratio = max(h_r / max(h_s, 1), w_r / max(w_s, 1))
        diag["pixel_scale_ratio"] = round(pixel_scale_ratio, 2)
        logger.info(
            f"Scale-Aware Matcher: pixel_scale_ratio={pixel_scale_ratio:.2f}x "
            f"(Src {w_s}x{h_s}, Ref {w_r}x{h_r})"
        )

        # ── Phase 1: GSD normalisation ────────────────────────────────────────
        # Scale source to reference pixel units so NCC and SIFT operate at same scale
        gsd_w = max(4, int(round(w_s / pixel_scale_ratio)))
        gsd_h = max(4, int(round(h_s / pixel_scale_ratio)))
        src_at_ref_gsd = cv2.resize(source_u8, (gsd_w, gsd_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Scale-Aware Matcher: src_at_ref_gsd={gsd_w}x{gsd_h}px")
        diag["src_at_ref_gsd_shape"] = (gsd_h, gsd_w)

        # ── Phase 2: Coarse NCC localisation ─────────────────────────────────
        rx1_n, ry1_n, rx2_n, ry2_n, ncc_conf = _ncc_localise(
            src_at_ref_gsd, reference_u8
        )
        diag["ncc_confidence"] = round(ncc_conf, 3)

        # Add 30% margin and clamp to image bounds
        roi_w = max(rx2_n - rx1_n, gsd_w)
        roi_h = max(ry2_n - ry1_n, gsd_h)
        margin_x = max(50, int(roi_w * 0.30))
        margin_y = max(50, int(roi_h * 0.30))
        rx1_n = max(0,   rx1_n - margin_x)
        ry1_n = max(0,   ry1_n - margin_y)
        rx2_n = min(w_r, rx2_n + margin_x)
        ry2_n = min(h_r, ry2_n + margin_y)
        diag["roi_native"] = (rx1_n, ry1_n, rx2_n, ry2_n)

        ref_crop_native = reference_u8[ry1_n:ry2_n, rx1_n:rx2_n]
        if ref_crop_native.size == 0:
            logger.info("Scale-Aware Matcher: ROI crop empty — falling back to full reference.")
            ref_crop_native = reference_u8
            rx1_n, ry1_n = 0, 0

        logger.info(
            f"Scale-Aware Matcher: Localised ROI "
            f"({ref_crop_native.shape[1]}x{ref_crop_native.shape[0]}px) "
            f"in full reference"
        )

        # ── Phase 3: Fine SIFT at common GSD ─────────────────────────────────
        # Scale ref_crop to the same GSD as src_at_ref_gsd, then resize both
        # to SIFT_MAX_DIM so that SIFT sees comparable feature sizes.
        src_fine, sf_src_fine = _resize_keep_ar(src_at_ref_gsd, _SIFT_MAX_DIM)

        # ref_crop is already at reference native resolution; resize for SIFT
        ref_crop_for_sift, sf_ref_crop_fine = _resize_keep_ar(ref_crop_native, _SIFT_MAX_DIM)
        diag["fine_src_shape"]  = src_fine.shape[:2]
        diag["fine_ref_shape"]  = ref_crop_for_sift.shape[:2]

        # Progressive SIFT with relaxing contrast threshold
        best_matches: tuple | None = None
        for contrast_t in [0.03, 0.015, 0.008]:
            pts_s, desc_s = _extract_sift(src_fine,           n=5000, contrast=contrast_t)
            pts_r, desc_r = _extract_sift(ref_crop_for_sift,  n=8000, contrast=contrast_t)
            if desc_s is None or desc_r is None or len(pts_s) == 0 or len(pts_r) == 0:
                continue
            for ratio in [0.78, 0.85]:
                idx1, idx2, confs = _match_flann(desc_s, desc_r, ratio=ratio)
                if len(idx1) >= 6:
                    best_matches = (pts_s, pts_r, idx1, idx2, confs)
                    break
            if best_matches is not None and len(best_matches[2]) >= 6:
                break

        if best_matches is None:
            logger.info("Scale-Aware Matcher: No SIFT matches at any contrast level.")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,),   dtype=np.float32),
                time.perf_counter() - t0,
                diag,
            )

        pts_s, pts_r, idx1, idx2, confs = best_matches
        n_matches = len(idx1)
        diag["fine_matches"] = n_matches
        logger.info(f"Scale-Aware Matcher: {n_matches} SIFT matches at GSD-normalised scale")

        # ── Phase 4: Coordinate remapping to native pixel frames ──────────────
        # src_fine coords → src_at_ref_gsd coords → source native coords
        #   src_fine coords ÷ sf_src_fine = src_at_ref_gsd coords
        #   src_at_ref_gsd coords × pixel_scale_ratio = source native coords
        src_pts_at_gsd   = pts_s[idx1] / sf_src_fine
        src_pts_native   = (src_pts_at_gsd * pixel_scale_ratio).astype(np.float32)

        # ref_crop_for_sift coords → ref_crop native coords → full reference native coords
        #   ref_crop_for_sift coords ÷ sf_ref_crop_fine = ref_crop native coords
        #   ref_crop native coords + (rx1_n, ry1_n) = full reference native coords
        ref_pts_crop     = pts_r[idx2] / sf_ref_crop_fine
        ref_pts_native   = (ref_pts_crop + np.array([rx1_n, ry1_n], dtype=np.float32)).astype(np.float32)

        # Sanity clamp to valid image bounds
        src_pts_native[:, 0] = np.clip(src_pts_native[:, 0], 0, w_s - 1)
        src_pts_native[:, 1] = np.clip(src_pts_native[:, 1], 0, h_s - 1)
        ref_pts_native[:, 0] = np.clip(ref_pts_native[:, 0], 0, w_r - 1)
        ref_pts_native[:, 1] = np.clip(ref_pts_native[:, 1], 0, h_r - 1)

        runtime = time.perf_counter() - t0
        logger.info(
            f"Scale-Aware Matcher: Completed in {runtime:.3f}s — "
            f"{n_matches} correspondences (NCC={ncc_conf:.3f})"
        )
        return src_pts_native, ref_pts_native, confs, runtime, diag
