#!/usr/bin/env python3

import numpy as np


def _preparse_particles(particles_list):
    """Parse each particle line once and precompute terms used in the Euler-distance test."""
    deg2rad = np.pi / 180.0

    number = len(particles_list)
    scores = np.empty(number, dtype=np.float64)
    xs = np.empty(number, dtype=np.float64)
    ys = np.empty(number, dtype=np.float64)

    cos_theta_half = np.empty(number, dtype=np.float64)
    sin_theta_half = np.empty(number, dtype=np.float64)
    cos_plus_half = np.empty(number, dtype=np.float64)   # cos((psi + phi) / 2)
    sin_plus_half = np.empty(number, dtype=np.float64)   # sin((psi + phi) / 2)
    cos_minus_half = np.empty(number, dtype=np.float64)  # cos((psi - phi) / 2)
    sin_minus_half = np.empty(number, dtype=np.float64)  # sin((psi - phi) / 2)

    for i in range(number):
        fields = particles_list[i].split('\t')

        # Original code uses fields[5] = ...theta,phi,psi...
        angle_values = fields[5].split('=', 1)[1].split(',')
        theta = float(angle_values[0]) * deg2rad
        phi = float(angle_values[1]) * deg2rad
        psi = float(angle_values[2]) * deg2rad

        # Original code uses fields[6] = ...x,y...
        center_values = fields[6].split('=', 1)[1].split(',')
        xs[i] = float(center_values[0])
        ys[i] = float(center_values[1])

        # Original code uses fields[7] = ...score...
        scores[i] = float(fields[7].split('=', 1)[1])

        theta_half = theta * 0.5
        plus_half = (psi + phi) * 0.5
        minus_half = (psi - phi) * 0.5

        cos_theta_half[i] = np.cos(theta_half)
        sin_theta_half[i] = np.sin(theta_half)
        cos_plus_half[i] = np.cos(plus_half)
        sin_plus_half[i] = np.sin(plus_half)
        cos_minus_half[i] = np.cos(minus_half)
        sin_minus_half[i] = np.sin(minus_half)

    return (
        scores, xs, ys,
        cos_theta_half, sin_theta_half,
        cos_plus_half, sin_plus_half,
        cos_minus_half, sin_minus_half,
    )


def remove_repeat_particles_from_one_micrograph(particles_list, center, euler_thres, score_threshold):
    """
    Faster drop-in replacement for the original function.

    The output order and keep/remove rules are intentionally kept the same as the
    original version. The main speedups are:
      1. parse each particle line only once;
      2. precompute repeated trigonometric terms;
      3. process the inner j-loop with NumPy arrays while preserving the same
         remove/keep decisions.
    """
    (
        scores, xs, ys,
        cos_theta_half, sin_theta_half,
        cos_plus_half, sin_plus_half,
        cos_minus_half, sin_minus_half,
    ) = _preparse_particles(particles_list)

    number = len(particles_list)
    removed = np.zeros(number, dtype=np.uint8)
    center2 = center * center
    rad2deg2 = 2.0 * 180.0 / np.pi

    for i in range(number):
        # Same as original: skip this i only if it was already removed before
        # entering the i-loop.
        if removed[i] == 1:
            continue

        score1 = scores[i]
        if score1 <= score_threshold:
            removed[i] = 1
            continue

        j_start = i + 1
        if j_start >= number:
            continue

        # These are exactly the j entries that would pass "if a[j] == 1: continue"
        # at the start of the original inner loop.
        active_j = np.nonzero(removed[j_start:number] == 0)[0] + j_start
        if active_j.size == 0:
            continue

        # Same threshold rule as original inner loop:
        # if score2 <= score_threshold: a[j] = 1; continue
        score_ok = scores[active_j] > score_threshold
        if not np.all(score_ok):
            removed[active_j[~score_ok]] = 1
        active_j = active_j[score_ok]
        if active_j.size == 0:
            continue

        # Same as original distance test, but using squared distance to avoid sqrt.
        # This is equivalent for center > 0. For center <= 0, the original condition
        # can never be true because distance is non-negative.
        if center <= 0:
            continue

        dx = xs[active_j] - xs[i]
        dy = ys[active_j] - ys[i]
        close_j = active_j[(dx * dx + dy * dy) < center2]
        if close_j.size == 0:
            continue

        ct_i = cos_theta_half[i]
        st_i = sin_theta_half[i]
        cp_i = cos_plus_half[i]
        sp_i = sin_plus_half[i]
        cm_i = cos_minus_half[i]
        sm_i = sin_minus_half[i]

        # Same formula/order as the original, but vectorized and with precomputed terms.
        r0 = (
            ct_i * cos_theta_half[close_j] * cp_i * cos_plus_half[close_j]
            + st_i * sin_theta_half[close_j] * cm_i * cos_minus_half[close_j]
            + st_i * sin_theta_half[close_j] * sm_i * sin_minus_half[close_j]
            + ct_i * cos_theta_half[close_j] * sp_i * sin_plus_half[close_j]
        )
        r0 = np.round(r0, 10)  # keep original floating-point treatment
        euler_dist = np.abs((np.arccos(r0) * rad2deg2 + 180.0) % 360.0 - 180.0)

        repeat_j = close_j[euler_dist < euler_thres]
        if repeat_j.size == 0:
            continue

        # Original rule for every repeated pair:
        #   if score1 > score2: remove j
        #   else: remove i
        remove_j = repeat_j[score1 > scores[repeat_j]]
        if remove_j.size > 0:
            removed[remove_j] = 1
        if np.any(score1 <= scores[repeat_j]):
            removed[i] = 1

    return [p for p,r in zip(particles_list, removed) if r==0]
