from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_FILES = BASE_DIR / "Project_files"
CURRENCY_DATA = BASE_DIR / "currency" / "data"
REFERENCE_CACHE = BASE_DIR / "reference_cache.npz"
TENSORFLOW_MODEL = BASE_DIR / "currency_authenticity_model.keras"
TENSORFLOW_MODEL_METADATA = BASE_DIR / "currency_authenticity_model.json"
SUPPORTED_DENOMS = ("10", "20", "50", "100", "200", "500", "2000")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_GENERATED_FEATURE_COUNT = 8
MIN_GENERATED_FEATURE_AVG = 0.14
DEFAULT_TENSORFLOW_THRESHOLDS = {
    "fake_reject_probability": 0.90,
    "fake_reject_margin": 0.20,
    "fake_review_probability": 0.70,
    "fake_review_margin": 0.10,
}


@dataclass(frozen=True)
class DenominationConfig:
    denomination: str
    resize: tuple[int, int]
    feature_dir: Path
    search_area_list: list[list[int]]
    feature_area_limits_list: list[list[int]]
    min_ssim_score_list: list[float]
    left_bleed_crop: tuple[int, int, int, int]
    right_bleed_crop: tuple[int, int, int, int]
    bleed_min: float
    bleed_max: float
    number_panel_crop: tuple[int, int, int, int]
    number_threshold_start: int


@dataclass(frozen=True)
class ReferenceSample:
    denomination: str
    kind: str
    path: Path
    vector: np.ndarray


@dataclass(frozen=True)
class FeatureTemplate:
    denomination: str
    path: Path
    image: np.ndarray
    gray: np.ndarray


CONFIGS = [
    DenominationConfig(
        denomination="Rs. 500",
        resize=(1167, 519),
        feature_dir=PROJECT_FILES / "Dataset" / "500_Features Dataset",
        search_area_list=[
            [200, 300, 200, 370],
            [1050, 1500, 300, 450],
            [100, 450, 20, 120],
            [690, 1050, 20, 120],
            [820, 1050, 350, 430],
            [700, 810, 330, 430],
            [400, 650, 0, 100],
        ],
        feature_area_limits_list=[
            [12000, 17000],
            [10000, 18000],
            [20000, 30000],
            [24000, 36000],
            [15000, 25000],
            [7000, 13000],
            [11000, 18000],
        ],
        min_ssim_score_list=[0.4, 0.4, 0.5, 0.4, 0.5, 0.45, 0.5],
        left_bleed_crop=(120, 240, 12, 35),
        right_bleed_crop=(120, 260, 1135, 1155),
        bleed_min=4.7,
        bleed_max=5.6,
        number_panel_crop=(410, 500, 700, 1080),
        number_threshold_start=95,
    ),
    DenominationConfig(
        denomination="Rs. 2000",
        resize=(1165, 455),
        feature_dir=PROJECT_FILES / "Dataset" / "2000_Features Dataset",
        search_area_list=[
            [200, 270, 160, 330],
            [1050, 1500, 250, 400],
            [50, 400, 0, 100],
            [750, 1050, 0, 100],
            [850, 1050, 280, 380],
            [700, 820, 290, 370],
            [400, 650, 0, 100],
        ],
        feature_area_limits_list=[
            [10000, 14000],
            [9000, 15000],
            [17000, 21500],
            [19000, 28000],
            [17500, 23000],
            [6500, 9000],
            [10000, 16000],
        ],
        min_ssim_score_list=[0.45, 0.4, 0.45, 0.45, 0.5, 0.4, 0.5],
        left_bleed_crop=(80, 230, 10, 30),
        right_bleed_crop=(90, 230, 1140, 1160),
        bleed_min=6.7,
        bleed_max=7.6,
        number_panel_crop=(360, 440, 760, 1080),
        number_threshold_start=90,
    ),
]


def analyze_currency(image: np.ndarray) -> dict:
    notebook_results = [run_notebook_pipeline(image, config) for config in CONFIGS]
    notebook_result = max(notebook_results, key=lambda item: (item["verified_count"], item["avg_feature_score"]))
    dataset_result = run_reference_dataset_pipeline(image)

    result = choose_final_result(notebook_result, dataset_result)
    accepted = result["accepted"]
    suspicious = result["suspicious"]
    confidence = result["confidence"]

    return {
        "accepted": accepted,
        "suspicious": suspicious,
        "denomination": result["denomination"],
        "confidence": confidence,
        "authenticity": confidence,
        "isKnownFake": result["is_fake_like"],
        "message": result["message"],
        "checks": result["checks"],
    }


def choose_final_result(notebook_result: dict, dataset_result: dict) -> dict:
    notebook_confidence = int(round((notebook_result["verified_count"] / 10) * 100))
    notebook_accepts = notebook_result["verified_count"] >= 8
    notebook_suspicious = not notebook_accepts and notebook_result["verified_count"] >= 4

    notebook_payload = {
        "accepted": notebook_accepts,
        "suspicious": notebook_suspicious,
        "denomination": notebook_result["denomination"],
        "confidence": notebook_confidence,
        "is_fake_like": False,
        "message": (
            f"{notebook_result['verified_count']} out of 10 original notebook features are verified."
            if notebook_accepts
            else f"Only {notebook_result['verified_count']} out of 10 original notebook features are verified."
        ),
        "checks": [
            {"title": "Original notebook pipeline", "passed": True, "warning": False, "detail": "Used ORB, SSIM, bleed-line, and number-panel checks from the existing notebooks."},
            *notebook_result["checks"],
        ],
    }

    if dataset_result is None:
        return notebook_payload

    dataset_checks = [
        {"title": "Generated notebook-style pipeline", "passed": True, "warning": False, "detail": "Used 10 local Python/OpenCV checks built from the provided denomination data."},
        *dataset_result["checks"],
    ]
    dataset_payload = {
        "accepted": dataset_result["accepted"],
        "suspicious": dataset_result["suspicious"],
        "denomination": dataset_result["denomination"],
        "confidence": dataset_result["confidence"],
        "is_fake_like": dataset_result["is_fake_like"],
        "message": dataset_result["message"],
        "checks": dataset_checks,
    }

    if dataset_result["is_fake_like"]:
        return dataset_payload

    if dataset_result.get("needs_review"):
        if notebook_accepts and notebook_result["denomination"] in {"Rs. 500", "Rs. 2000"}:
            notebook_payload["confidence"] = max(notebook_confidence, dataset_result["confidence"])
            notebook_payload["message"] = (
                f"{notebook_result['verified_count']} out of 10 original notebook features are verified for "
                f"{notebook_result['denomination']}; generated checks need review."
            )
            notebook_payload["checks"].extend(dataset_checks)
            return notebook_payload
        if notebook_accepts and notebook_result["denomination"] == dataset_result["denomination"]:
            dataset_payload["checks"].extend(notebook_payload["checks"])
        return dataset_payload

    if notebook_accepts and notebook_result["denomination"] == dataset_result["denomination"]:
        combined_confidence = max(notebook_confidence, dataset_result["confidence"])
        notebook_payload["confidence"] = combined_confidence
        notebook_payload["message"] = f"{notebook_result['verified_count']} out of 10 original notebook features are verified; local OpenCV checks also matched {dataset_result['denomination']}."
        notebook_payload["checks"].extend(dataset_checks)
        return notebook_payload

    if dataset_result["denomination"] not in {"Rs. 500", "Rs. 2000"}:
        return dataset_payload

    if not notebook_accepts:
        notebook_payload["checks"].extend(dataset_checks)
        return notebook_payload

    return notebook_payload


def run_reference_dataset_pipeline(image: np.ndarray) -> dict | None:
    references = load_reference_samples()
    if not references:
        return None

    normalized = normalize_note_image(image)
    query_vector = image_vector(normalized)

    fast_scored = [(cosine_similarity(query_vector, sample.vector), sample) for sample in references]
    fast_scored.sort(key=lambda item: item[0], reverse=True)
    scored = fast_scored[:80]

    real_scores = [item for item in scored if item[1].kind == "real"]
    fake_scores = [item for item in scored if item[1].kind == "fake"]
    best_real_score, best_real = real_scores[0] if real_scores else (0.0, None)
    best_fake_score, best_fake = fake_scores[0] if fake_scores else (0.0, None)
    top_real_mean = mean_score(real_scores, 8)
    top_fake_mean = mean_score(fake_scores, 8)
    fake_neighbor_ratio = sum(1 for _, sample in scored[:20] if sample.kind == "fake") / max(1, min(20, len(scored)))
    denom_scores: dict[str, list[float]] = {}
    for score, sample in scored:
        if sample.kind == "real":
            denom_scores.setdefault(sample.denomination, []).append(score)

    best_denom = None
    best_denom_score = 0.0
    for denomination, scores in denom_scores.items():
        top_scores = scores[: min(8, len(scores))]
        mean_top_score = float(sum(top_scores) / len(top_scores))
        if mean_top_score > best_denom_score:
            best_denom = denomination
            best_denom_score = mean_top_score

    if best_denom is None:
        return None

    authenticity = best_real_score - best_fake_score
    strong_genuine_match = best_real is not None and best_real_score >= 0.88 and authenticity >= 0.12
    if strong_genuine_match:
        best_denom = best_real.denomination
        best_denom_score = max(best_denom_score, best_real_score)

    visual_denom, visual_feature_count, visual_avg_score = estimate_visual_denomination(image)
    denomination_consistent = visual_denom is None or visual_denom == best_denom
    generated_checks, generated_verified_count, generated_avg_score = run_generated_notebook_style_checks(image, best_denom)
    tensorflow_result = run_tensorflow_authenticator(image)
    reference_margin = top_real_mean - top_fake_mean
    nearest_fake_hit = best_fake_score >= 0.82 and authenticity < 0.03
    neighborhood_fake_hit = (
        not strong_genuine_match
        and top_fake_mean >= 0.72
        and top_fake_mean > top_real_mean
        and fake_neighbor_ratio >= 0.45
    )
    fake_reference_match = (nearest_fake_hit or neighborhood_fake_hit) and not strong_genuine_match
    generated_features_pass = generated_verified_count >= MIN_GENERATED_FEATURE_COUNT and generated_avg_score >= MIN_GENERATED_FEATURE_AVG
    strong_genuine_pass = strong_genuine_match and generated_verified_count >= 5 and generated_avg_score >= 0.10
    tensorflow_fake_hit = is_tensorflow_fake_hit(tensorflow_result) and should_trust_tensorflow_fake(
        tensorflow_result,
        generated_verified_count,
        best_real_score,
        authenticity,
        reference_margin,
        fake_reference_match,
    )
    tensorflow_review = is_tensorflow_review(tensorflow_result, tensorflow_fake_hit)
    is_fake_like = fake_reference_match or tensorflow_fake_hit
    accepted = (not is_fake_like) and (not tensorflow_review) and best_real_score >= 0.50 and best_denom_score >= 0.46 and (generated_features_pass or strong_genuine_pass)
    suspicious = (not accepted) and (not is_fake_like) and (
        best_real_score >= 0.42
        or best_denom_score >= 0.40
        or generated_verified_count >= 4
        or tensorflow_review
        or (fake_reference_match and generated_features_pass)
    )
    confidence = int(round(max(generated_verified_count / 10, min(1.0, best_denom_score)) * 100))

    if accepted and strong_genuine_pass and not generated_features_pass:
        message = f"Strong genuine-reference match found for Rs. {best_denom}; {generated_verified_count} out of 10 visual checks also matched."
    elif accepted:
        message = f"{generated_verified_count} out of 10 generated notebook-style features are verified for Rs. {best_denom}."
    elif is_fake_like:
        if tensorflow_fake_hit:
            message = "TensorFlow authenticity model strongly predicts fake and supporting checks did not clear the note."
        else:
            message = (
                "Fake-note reference similarity is stronger than genuine-note similarity, "
                f"even though {generated_verified_count} visual denomination checks matched."
            )
    elif tensorflow_review:
        message = f"{generated_verified_count} out of 10 visual checks matched, but TensorFlow fake probability is high enough for manual review."
    elif not generated_features_pass:
        message = (
            f"{generated_verified_count} out of 10 generated notebook-style features passed, "
            f"but minimum {MIN_GENERATED_FEATURE_COUNT} checks and {MIN_GENERATED_FEATURE_AVG * 100:.0f}% average feature score are required."
        )
        if fake_reference_match:
            message += " Fake-note reference similarity also failed."
    elif fake_reference_match:
        message = f"{generated_verified_count} out of 10 visual denomination checks matched, but fake-note reference similarity needs review."
    else:
        message = f"Only {generated_verified_count} out of 10 generated notebook-style features are verified."

    checks = [
        make_check(
            "Denomination match",
            best_denom_score >= 0.46,
            f"Best match Rs. {best_denom} with score {best_denom_score * 100:.0f}%.",
        ),
        make_check(
            "Strong genuine reference",
            strong_genuine_match,
            f"Closest real score {best_real_score * 100:.0f}% versus closest fake score {best_fake_score * 100:.0f}%.",
        ),
        make_check(
            "Visual denomination consistency",
            denomination_consistent,
            f"Visual feature templates matched Rs. {visual_denom or '-'} with {visual_feature_count}/7 denomination features and {visual_avg_score * 100:.0f}% average score.",
        ),
        make_check(
            "Generated feature average",
            generated_avg_score >= MIN_GENERATED_FEATURE_AVG,
            f"Average generated feature score {generated_avg_score * 100:.0f}%; minimum {MIN_GENERATED_FEATURE_AVG * 100:.0f}%.",
        ),
        make_tensorflow_check(tensorflow_result, tensorflow_fake_hit),
        make_check(
            "Nearest genuine note",
            best_real_score >= 0.50,
            f"Closest real reference {best_real.path.parent.name if best_real else '-'} scored {best_real_score * 100:.0f}%.",
        ),
        make_check(
            "Nearest fake-note comparison",
            not fake_reference_match,
            f"Closest fake reference {best_fake.path.parent.name if best_fake else '-'} scored {best_fake_score * 100:.0f}%.",
        ),
        make_check(
            "Reference neighborhood",
            not neighborhood_fake_hit,
            f"Top real average {top_real_mean * 100:.0f}%, top fake average {top_fake_mean * 100:.0f}%, fake-neighbor ratio {fake_neighbor_ratio * 100:.0f}%.",
        ),
        make_check(
            "Authenticity margin",
            authenticity >= 0.04 or reference_margin >= 0.025,
            f"Nearest margin {authenticity * 100:.0f}%, neighborhood margin {reference_margin * 100:.0f}%.",
        ),
        *generated_checks,
    ]

    return {
        "accepted": accepted,
        "suspicious": suspicious,
        "denomination": f"Rs. {best_denom}",
        "confidence": confidence,
        "is_fake_like": is_fake_like,
        "needs_review": tensorflow_review,
        "message": message,
        "checks": checks,
    }


def run_generated_notebook_style_checks(image: np.ndarray, denomination: str) -> tuple[list[dict], int, float]:
    feature_scores = score_selected_denomination_features(image, denomination)
    normalized = cv2.resize(crop_foreground(image), (1167, 519))
    gray = cv2.cvtColor(cv2.GaussianBlur(normalized, (5, 5), 0), cv2.COLOR_BGR2GRAY)

    checks = []
    verified_count = 0
    for index, score in enumerate(feature_scores, start=1):
        passed = score >= 0.18
        verified_count += int(passed)
        checks.append(make_check(f"Generated Feature {index} ORB/SSIM", passed, f"Feature score {score * 100:.0f}%; minimum 18%."))

    left_score = edge_band_score(gray[:, :80])
    right_score = edge_band_score(gray[:, -80:])
    texture_score = serial_texture_score(gray)
    extra_checks = [
        make_check("Generated Feature 8 left edge/bleed signal", left_score >= 0.005, f"Edge signal {left_score * 100:.1f}%; minimum 0.5%."),
        make_check("Generated Feature 9 right edge/bleed signal", right_score >= 0.005, f"Edge signal {right_score * 100:.1f}%; minimum 0.5%."),
        make_check("Generated Feature 10 serial/print texture", texture_score >= 0.02, f"Texture signal {texture_score * 100:.1f}%; minimum 2%."),
    ]
    for check in extra_checks:
        verified_count += int(check["passed"])
        checks.append(check)

    avg_score = float((sum(feature_scores) + left_score + right_score + texture_score) / 10)
    return checks, verified_count, avg_score


def mean_score(scored: list[tuple[float, ReferenceSample]], limit: int) -> float:
    if not scored:
        return 0.0
    scores = [score for score, _ in scored[:limit]]
    return float(sum(scores) / len(scores))


@lru_cache(maxsize=1)
def load_tensorflow_authenticator():
    if not TENSORFLOW_MODEL.exists():
        return None
    try:
        import tensorflow as tf
    except ImportError:
        return None

    model = tf.keras.models.load_model(TENSORFLOW_MODEL)
    if TENSORFLOW_MODEL_METADATA.exists():
        metadata = json.loads(TENSORFLOW_MODEL_METADATA.read_text(encoding="utf-8"))
    else:
        metadata = {"class_names": ["fake", "real"], "image_size": [224, 224], "thresholds": DEFAULT_TENSORFLOW_THRESHOLDS}
    return model, metadata


def run_tensorflow_authenticator(image: np.ndarray) -> dict | None:
    loaded = load_tensorflow_authenticator()
    if loaded is None:
        return None

    model, metadata = loaded
    image_size = tuple(metadata.get("image_size", [224, 224]))
    class_names = list(metadata.get("class_names", ["fake", "real"]))

    batch = np.stack([prepare_tensorflow_view(view, image_size) for view in tensorflow_input_views(image)])
    predictions = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape[1] == 1:
        fake_predictions = predictions[:, 0]
        real_predictions = 1.0 - fake_predictions
    else:
        fake_index = class_names.index("fake") if "fake" in class_names else 0
        real_index = class_names.index("real") if "real" in class_names else min(1, predictions.shape[1] - 1)
        fake_predictions = predictions[:, fake_index]
        real_predictions = predictions[:, real_index]

    fake_probability = float(np.mean(fake_predictions))
    real_probability = float(np.mean(real_predictions))
    thresholds = tensorflow_thresholds(metadata.get("thresholds", {}))

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "fake_probability_max": float(np.max(fake_predictions)),
        "real_probability_max": float(np.max(real_predictions)),
        "view_count": int(len(fake_predictions)),
        "thresholds": thresholds,
    }


def tensorflow_thresholds(metadata_thresholds: dict | None) -> dict:
    configured = {**DEFAULT_TENSORFLOW_THRESHOLDS, **(metadata_thresholds or {})}
    # Runtime floors keep older metadata from making TensorFlow too aggressive.
    return {
        "fake_reject_probability": max(configured["fake_reject_probability"], DEFAULT_TENSORFLOW_THRESHOLDS["fake_reject_probability"]),
        "fake_reject_margin": max(configured["fake_reject_margin"], DEFAULT_TENSORFLOW_THRESHOLDS["fake_reject_margin"]),
        "fake_review_probability": max(configured["fake_review_probability"], DEFAULT_TENSORFLOW_THRESHOLDS["fake_review_probability"]),
        "fake_review_margin": max(configured["fake_review_margin"], DEFAULT_TENSORFLOW_THRESHOLDS["fake_review_margin"]),
    }


def tensorflow_input_views(image: np.ndarray) -> list[np.ndarray]:
    views = [crop_foreground(image), crop_largest_note_region(image), image]
    unique_views = []
    seen_shapes = set()
    for view in views:
        if view.size == 0:
            continue
        shape_key = tuple(view.shape[:2])
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        unique_views.append(view)
    return unique_views or [image]


def prepare_tensorflow_view(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    return resized.astype(np.float32)


def is_tensorflow_fake_hit(result: dict | None) -> bool:
    if result is None:
        return False
    thresholds = result.get("thresholds", DEFAULT_TENSORFLOW_THRESHOLDS)
    fake_probability = result["fake_probability"]
    real_probability = result["real_probability"]
    return (
        fake_probability >= thresholds["fake_reject_probability"]
        and fake_probability >= real_probability + thresholds["fake_reject_margin"]
    )


def should_trust_tensorflow_fake(
    result: dict | None,
    generated_verified_count: int,
    best_real_score: float,
    authenticity: float,
    reference_margin: float,
    fake_reference_match: bool,
) -> bool:
    if result is None:
        return False

    fake_probability = result["fake_probability"]
    visual_supports_note = generated_verified_count >= 6 and best_real_score >= 0.60
    references_are_ambiguous = authenticity >= -0.03 and reference_margin >= -0.03
    very_strong_tensorflow_fake = fake_probability >= 0.96

    if visual_supports_note and references_are_ambiguous and not fake_reference_match and not very_strong_tensorflow_fake:
        return False

    return True


def is_tensorflow_review(result: dict | None, fake_hit: bool = False) -> bool:
    if result is None or fake_hit:
        return False
    thresholds = result.get("thresholds", DEFAULT_TENSORFLOW_THRESHOLDS)
    fake_probability = result["fake_probability"]
    real_probability = result["real_probability"]
    return (
        fake_probability >= thresholds["fake_review_probability"]
        and fake_probability >= real_probability + thresholds["fake_review_margin"]
    )


def make_tensorflow_check(result: dict | None, fake_hit: bool) -> dict:
    if result is None:
        return {
            "title": "TensorFlow authenticity model",
            "passed": True,
            "warning": False,
            "detail": "No trained TensorFlow model found; skipped optional neural classifier.",
        }
    return make_check(
        "TensorFlow authenticity model",
        not fake_hit,
        (
            f"Real probability {result['real_probability'] * 100:.0f}%, "
            f"fake probability {result['fake_probability'] * 100:.0f}% "
            f"across {result.get('view_count', 1)} image views."
        ),
    )


def estimate_visual_denomination(image: np.ndarray) -> tuple[str | None, int, float]:
    best_denom = None
    best_count = 0
    best_avg = 0.0
    for denomination in SUPPORTED_DENOMS:
        scores = score_selected_denomination_features(image, denomination)
        count = sum(score >= 0.18 for score in scores)
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        if (count, avg_score) > (best_count, best_avg):
            best_denom = denomination
            best_count = count
            best_avg = avg_score
    return best_denom, best_count, best_avg


def score_selected_denomination_features(image: np.ndarray, denomination: str) -> list[float]:
    templates = [template for template in load_feature_templates() if template.denomination == denomination]
    if not templates:
        return [0.0] * 7
    normalized = cv2.resize(crop_foreground(image), (1167, 519))
    blur_query = cv2.GaussianBlur(normalized, (5, 5), 0)
    gray_query = cv2.cvtColor(blur_query, cv2.COLOR_BGR2GRAY)
    groups = split_templates_into_feature_groups(templates, 7)
    scores = []
    for group in groups:
        group_scores = [match_feature_template(template, blur_query, gray_query) for template in group[:2]]
        scores.append(max(group_scores) if group_scores else 0.0)
    return scores[:7] + [0.0] * max(0, 7 - len(scores))


def split_templates_into_feature_groups(templates: list[FeatureTemplate], group_count: int) -> list[list[FeatureTemplate]]:
    groups = [[] for _ in range(group_count)]
    for index, template in enumerate(templates):
        groups[index % group_count].append(template)
    return groups


def edge_band_score(gray_band: np.ndarray) -> float:
    if gray_band.size == 0:
        return 0.0
    edges = cv2.Canny(gray_band, 60, 150)
    return float(np.count_nonzero(edges) / edges.size)


def serial_texture_score(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    crop = gray[int(h * 0.50) : int(h * 0.92), int(w * 0.50) : int(w * 0.96)]
    if crop.size == 0:
        return 0.0
    edges = cv2.Canny(crop, 60, 150)
    return float(np.count_nonzero(edges) / edges.size)


@lru_cache(maxsize=1)
def load_feature_templates() -> tuple[FeatureTemplate, ...]:
    templates: list[FeatureTemplate] = []
    root = BASE_DIR / "Features" / "Features"
    if not root.exists():
        return tuple()

    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or not directory.name.endswith("_Features"):
            continue
        denomination = directory.name.replace("_Features", "")
        if denomination not in SUPPORTED_DENOMS:
            continue
        files = sorted([path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name)
        step = max(1, len(files) // 12)
        for path in files[::step][:12]:
            image = read_image(path)
            if image is None or min(image.shape[:2]) < 35:
                continue
            image = cv2.GaussianBlur(image, (5, 5), 0)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            templates.append(FeatureTemplate(denomination=denomination, path=path, image=image, gray=gray))
    return tuple(templates)


def match_feature_template(template: FeatureTemplate, blur_query: np.ndarray, gray_query: np.ndarray) -> float:
    match = compute_orb(template.gray, gray_query)
    if match is None:
        return 0.0
    dst, dst_pts = match
    x, y, w, h = cv2.boundingRect(dst)
    if w * h < 900 or w * h > blur_query.shape[0] * blur_query.shape[1] * 0.35:
        x, y, w, h = cv2.boundingRect(dst_pts)
    if w <= 0 or h <= 0:
        return 0.0
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(blur_query.shape[1], x + w)
    y2 = min(blur_query.shape[0], y + h)
    crop = blur_query[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return calculate_ssim(template.image, crop)


@lru_cache(maxsize=1)
def load_reference_samples() -> tuple[ReferenceSample, ...]:
    cached_samples = load_reference_cache()
    if cached_samples:
        return cached_samples

    samples: list[ReferenceSample] = []
    for kind, denomination, path in iter_reference_paths():
        image = read_image(path)
        if image is None:
            continue
        normalized = normalize_note_image(image)
        samples.append(
            ReferenceSample(
                denomination=denomination,
                kind=kind,
                path=path,
                vector=image_vector(normalized),
            )
        )
    return tuple(samples)


def load_reference_cache() -> tuple[ReferenceSample, ...]:
    if not REFERENCE_CACHE.exists():
        return tuple()
    cache = np.load(REFERENCE_CACHE, allow_pickle=False)
    denominations = cache["denominations"]
    kinds = cache["kinds"]
    paths = cache["paths"]
    vectors = cache["vectors"]
    samples = []
    for index in range(len(denominations)):
        path = BASE_DIR.joinpath(*str(paths[index]).replace("\\", "/").split("/"))
        samples.append(
            ReferenceSample(
                denomination=str(denominations[index]),
                kind=str(kinds[index]),
                path=path,
                vector=vectors[index],
            )
        )
    return tuple(samples)


def iter_reference_paths() -> list[tuple[str, str, Path]]:
    candidates: list[tuple[str, str, Path]] = []
    dataset_roots = [CURRENCY_DATA]

    for root in dataset_roots:
        for kind in ("real", "fake"):
            kind_dir = root / kind
            if not kind_dir.exists():
                continue
            for denom_dir in sorted(kind_dir.iterdir(), key=lambda item: item.name):
                if not denom_dir.is_dir() or denom_dir.name not in SUPPORTED_DENOMS:
                    continue
                files = sorted(
                    [path for path in denom_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS],
                    key=lambda path: path.name,
                )
                limit = 28 if kind == "real" else 22
                step = max(1, len(files) // limit)
                for path in files[::step][:limit]:
                    candidates.append((kind, denom_dir.name, path))

    legacy_dirs = [
        ("real", "500", PROJECT_FILES / "Dataset" / "500_dataset"),
        ("real", "2000", PROJECT_FILES / "Dataset" / "2000_dataset"),
        ("fake", "500", PROJECT_FILES / "Fake Notes" / "500"),
        ("fake", "2000", PROJECT_FILES / "Fake Notes" / "2000"),
    ]
    for kind, denomination, directory in legacy_dirs:
        if not directory.exists():
            continue
        files = sorted([path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name)
        for path in files[:28]:
            candidates.append((kind, denomination, path))

    return candidates


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize_note_image(image: np.ndarray) -> np.ndarray:
    crop = crop_foreground(image)
    resized = cv2.resize(crop, (420, 180))
    return cv2.GaussianBlur(resized, (3, 3), 0)


def crop_largest_note_region(image: np.ndarray) -> np.ndarray:
    image = crop_foreground(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    image_area = image.shape[0] * image.shape[1]
    best_rect = None
    best_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if area > best_area and area > image_area * 0.08 and 1.6 <= aspect <= 4.4:
            best_rect = (x, y, w, h)
            best_area = area

    if best_rect is None:
        return image

    x, y, w, h = best_rect
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.04)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    return image[y1:y2, x1:x2]


def crop_foreground(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 18) & (value > 40)) | ((gray > 25) & (gray < 235))
    points = cv2.findNonZero(mask.astype(np.uint8) * 255)
    if points is None:
        return image
    x, y, w, h = cv2.boundingRect(points)
    if w * h < image.shape[0] * image.shape[1] * 0.08:
        return image
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.04)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    return image[y1:y2, x1:x2]


def image_vector(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [10, 8, 6], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-9)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (42, 18)).astype(np.float32).flatten() / 255.0
    small = small - small.mean()
    small = small / (np.linalg.norm(small) + 1e-9)
    return np.concatenate([hist.astype(np.float32), small.astype(np.float32)])


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    score = float(np.dot(left, right) / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9))
    return (score + 1.0) / 2.0


def run_notebook_pipeline(image: np.ndarray, config: DenominationConfig) -> dict:
    test_img = cv2.resize(image, config.resize)
    blur_test_img = cv2.GaussianBlur(test_img, (5, 5), 0)
    gray_test_image = cv2.cvtColor(blur_test_img, cv2.COLOR_BGR2GRAY)

    feature_checks, avg_scores = test_features_1_to_7(test_img, blur_test_img, gray_test_image, config)
    left_line_count = count_bleed_lines(test_img, config.left_bleed_crop)
    right_line_count = count_bleed_lines(test_img, config.right_bleed_crop)
    number_panel_passed = test_number_panel(test_img, gray_test_image, config)

    checks = []
    verified_count = 0
    for check in feature_checks:
        verified_count += int(check["passed"])
        checks.append(check)

    left_passed = config.bleed_min <= left_line_count <= config.bleed_max
    right_passed = config.bleed_min <= right_line_count <= config.bleed_max
    verified_count += int(left_passed)
    verified_count += int(right_passed)
    verified_count += int(number_panel_passed)

    checks.extend(
        [
            make_check("Feature 8 left bleed lines", left_passed, f"Average count {left_line_count:.2f}; expected {config.bleed_min:.1f}-{config.bleed_max:.1f}."),
            make_check("Feature 9 right bleed lines", right_passed, f"Average count {right_line_count:.2f}; expected {config.bleed_min:.1f}-{config.bleed_max:.1f}."),
            make_check("Feature 10 number panel", number_panel_passed, "9 serial-number characters detected." if number_panel_passed else "9 serial-number characters were not detected."),
        ]
    )

    return {
        "denomination": config.denomination,
        "verified_count": verified_count,
        "avg_feature_score": float(sum(avg_scores) / max(len(avg_scores), 1)),
        "checks": checks,
    }


def test_features_1_to_7(test_img: np.ndarray, blur_test_img: np.ndarray, gray_test_image: np.ndarray, config: DenominationConfig) -> tuple[list[dict], list[float]]:
    checks = []
    avg_scores = []

    for feature_index in range(7):
        score_set = []
        max_score = -1.0

        for template_index in range(6):
            template_path = config.feature_dir / f"Feature {feature_index + 1}" / f"{template_index + 1}.jpg"
            template_img = cv2.imread(str(template_path))
            if template_img is None:
                continue

            template_img_blur = cv2.GaussianBlur(template_img, (5, 5), 0)
            template_img_gray = cv2.cvtColor(template_img_blur, cv2.COLOR_BGR2GRAY)
            test_img_mask = gray_test_image.copy()

            search_area = config.search_area_list[feature_index]
            test_img_mask[:, : search_area[0]] = 0
            test_img_mask[:, search_area[1] :] = 0
            test_img_mask[: search_area[2], :] = 0
            test_img_mask[search_area[3] :, :] = 0

            match = compute_orb(template_img_gray, test_img_mask)
            if match is None:
                continue
            dst, dst_pts = match

            x, y, w, h = cv2.boundingRect(dst)
            min_area, max_area = config.feature_area_limits_list[feature_index]
            feature_area = w * h

            if feature_area < min_area or feature_area > max_area:
                x, y, w, h = cv2.boundingRect(dst_pts)
                feature_area = w * h
                if feature_area < min_area or feature_area > max_area:
                    continue

            crop_img = blur_test_img[y : y + h, x : x + w]
            if crop_img.size == 0:
                continue

            score = calculate_ssim(template_img_blur, crop_img)
            score_set.append(score)
            max_score = max(max_score, score)

        avg_score = sum(score_set) / len(score_set) if score_set else 0.0
        avg_scores.append(avg_score)
        min_allowed = config.min_ssim_score_list[feature_index]
        passed = avg_score >= min_allowed or max_score >= 0.79
        checks.append(
            make_check(
                f"Feature {feature_index + 1} ORB/SSIM",
                passed,
                f"Average SSIM {avg_score:.3f}, max SSIM {max_score:.3f}, minimum {min_allowed:.2f}.",
            )
        )

    return checks, avg_scores


def calculate_ssim(template_img: np.ndarray, query_img: np.ndarray) -> float:
    min_w = min(template_img.shape[1], query_img.shape[1])
    min_h = min(template_img.shape[0], query_img.shape[0])
    img1 = cv2.resize(template_img, (min_w, min_h))
    img2 = cv2.resize(query_img, (min_w, min_h))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return structural_similarity(img1, img2)


def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(img1, kernel, sigma)
    mu2 = cv2.GaussianBlur(img2, kernel, sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu1_mu2
    score = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(score.mean())


def compute_orb(template_img: np.ndarray, query_img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    orb = cv2.ORB_create(700, 1.2, 8, 15)
    kpts1, descs1 = orb.detectAndCompute(template_img, None)
    kpts2, descs2 = orb.detectAndCompute(query_img, None)
    if descs1 is None or descs2 is None or len(descs1) < 4 or len(descs2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descs1, descs2), key=lambda x: x.distance)
    if len(matches) < 4:
        return None

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if matrix is None:
        return None

    h, w = template_img.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)
    return dst, dst_pts


def count_bleed_lines(test_img: np.ndarray, crop_box: tuple[int, int, int, int]) -> float:
    y1, y2, x1, x2 = crop_box
    crop = test_img[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    result = []
    width = thresh.shape[1]
    for j in range(width):
        col = thresh[:, j : j + 1]
        count = 0
        for i in range(len(col) - 1):
            pixel1_value = col[i][0]
            pixel2_value = col[i + 1][0]
            if pixel1_value != 0 and pixel1_value != 255:
                pixel1_value = 255
            if pixel2_value != 0 and pixel2_value != 255:
                pixel2_value = 255
            if pixel1_value == 255 and pixel2_value == 0:
                count += 1
        if 0 < count < 10:
            result.append(count)

    return float(sum(result) / len(result)) if result else -1.0


def test_number_panel(test_img: np.ndarray, gray_test_image: np.ndarray, config: DenominationConfig) -> bool:
    y1, y2, x1, x2 = config.number_panel_crop
    crop = gray_test_image[y1:y2, x1:x2]
    crop_bgr = test_img[y1:y2, x1:x2]
    if crop.size == 0 or crop_bgr.size == 0:
        return False

    count = 0
    for thresh_value in range(config.number_threshold_start, 155, 5):
        _, thresh = cv2.threshold(crop, thresh_value, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_and(crop, crop, mask=thresh)
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        h_img, _ = img.shape[:2]
        bounding_rect_list = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x != 0:
                bounding_rect_list.append([x, y, w, h])

        bounding_rect_list.sort()
        res_list = [rect for rect in bounding_rect_list if rect[2] * rect[3] > 150]

        i = 0
        while i < len(res_list):
            x, _, w, _ = res_list[i]
            j = i + 1
            while j < len(res_list):
                x0, _, w0, _ = res_list[j]
                if (x + w) >= x0 + w0:
                    res_list.pop(j)
                else:
                    break
            i += 1

        i = 0
        while i < len(res_list):
            _, y, _, h = res_list[i]
            if (h_img - (y + h)) > 40 or h < 17:
                res_list.pop(i)
            else:
                i += 1

        if len(res_list) == 9:
            count += 1
        if count == 3:
            break

    return count > 0


def make_check(title: str, passed: bool, detail: str) -> dict:
    return {"title": title, "passed": bool(passed), "warning": False, "detail": detail}
