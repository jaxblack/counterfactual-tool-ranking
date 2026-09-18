"""Paired policy contrasts with explicit sharp bounds for unsupported actions."""

import numpy as np


def disagreement_fallback(target: list[int], baseline: list[int], probabilities: list[np.ndarray]) -> list[int]:
    return [
        proposed if proposed == reference or (distribution[proposed] > 0 and distribution[reference] > 0) else reference
        for proposed, reference, distribution in zip(target, baseline, probabilities, strict=True)
    ]


def contrast_rows(probabilities, chosen, rewards, predictions, target, baseline, lower=-1.1, upper=1.0):
    if not lower < upper:
        raise ValueError("invalid outcome bounds")
    left, right, sampling_ranges = [], [], []
    unsupported_disagreements = 0
    disagreements = 0
    shared_unsupported = 0
    for distribution, action, reward, estimate, proposed, reference in zip(
        probabilities, chosen, rewards, predictions, target, baseline, strict=True,
    ):
        distribution = np.asarray(distribution, dtype=float)
        estimate = np.asarray(estimate, dtype=float)
        if (
            distribution.shape != estimate.shape or distribution.ndim != 1
            or not np.isclose(distribution.sum(), 1.0) or np.any(distribution < 0)
            or not np.all(np.isfinite(distribution)) or not np.all(np.isfinite(estimate))
            or not 0 <= action < len(distribution) or distribution[action] <= 0
            or not 0 <= proposed < len(distribution) or not 0 <= reference < len(distribution)
            or not lower <= reward <= upper
        ):
            raise ValueError("invalid logged contrast row")
        if proposed == reference:
            shared_unsupported += int(distribution[proposed] == 0)
            left.append(0.0)
            right.append(0.0)
            sampling_ranges.append(0.0)
            continue
        disagreements += 1
        estimates = np.clip(estimate, lower, upper)
        coefficients = np.zeros(len(distribution))
        coefficients[proposed], coefficients[reference] = 1, -1
        support = distribution > 0
        observed_term = float(np.dot(coefficients[support], estimates[support]))
        observed_term += coefficients[action] * (reward - estimates[action]) / distribution[action]
        missing = coefficients[~support]
        missing_lower = float(np.where(missing >= 0, missing * lower, missing * upper).sum())
        missing_upper = float(np.where(missing >= 0, missing * upper, missing * lower).sum())
        left.append(observed_term + missing_lower)
        right.append(observed_term + missing_upper)
        unsupported_disagreements += int(np.any(missing != 0))
        correction_low = coefficients[support] * (lower - estimates[support]) / distribution[support]
        correction_high = coefficients[support] * (upper - estimates[support]) / distribution[support]
        sampling_ranges.append(float(np.maximum(correction_low, correction_high).max() - np.minimum(correction_low, correction_high).min()))
    if not left:
        raise ValueError("contrast requires at least one decision")
    return {
        "lower_rows": np.array(left), "upper_rows": np.array(right), "sampling_ranges": np.array(sampling_ranges),
        "disagreements": disagreements, "unsupported_disagreements": unsupported_disagreements,
        "shared_unsupported": shared_unsupported,
    }


def evaluate_contrast(probabilities, chosen, rewards, predictions, target, baseline, lower=-1.1, upper=1.0, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rows = contrast_rows(probabilities, chosen, rewards, predictions, target, baseline, lower, upper)
    total = len(rows["lower_rows"])
    radius = float(np.sqrt(np.log(2 / alpha) * np.square(rows["sampling_ranges"]).sum() / (2 * total**2)))
    partial_lower = float(rows["lower_rows"].mean())
    partial_upper = float(rows["upper_rows"].mean())
    return {
        "tasks": total, "disagreements": rows["disagreements"],
        "unsupported_disagreements": rows["unsupported_disagreements"],
        "shared_unsupported": rows["shared_unsupported"],
        "point_identified": rows["unsupported_disagreements"] == 0,
        "estimated_identification_interval": [partial_lower, partial_upper],
        "identification_width": partial_upper - partial_lower,
        "conditional_hoeffding_interval": [partial_lower - radius, partial_upper + radius],
        "sampling_radius": radius, "alpha": alpha,
        "confidence_scope": "independent randomized logging, frozen policies/nuisance, fixed observed contexts; not future-distribution certification",
    }