"""The zone prediction pipeline.

YOLOv8 finds the zone circle in a screenshot, SIFT + homography aligns that
screenshot against one stitched template containing every map, and a per-map
per-phase Keras model predicts where the next circle lands.

Only Erangel and Miramar are supported. The template image still contains all
four maps -- the quadrant layout defines the coordinate system, so it must not
change -- but screenshots that land in the Vikendi or Sanhok quadrants are
rejected with a clear message instead of being silently mispredicted.
"""
import json
import os
import pickle

import cv2
import numpy as np
import keras
from ultralytics import YOLO

from config import PATH_PREFIX, SUPPORTED_MAPS, RETIRED_MAPS
from errors import PredictionError

# Quadrant boundaries within Map/combine_new.png.
QUADRANT_X = 1280
QUADRANT_Y = 1440

# Map code -> key in constants.json. ("erangle" is the historical spelling.)
MAP_CONSTANT_KEYS = {"e": "erangle", "m": "miramar"}

MODEL_NAMES = [f"{m}{z}{z + 1}" for m in ("e", "m") for z in range(1, 8)]

sift = None
flann = None
map_constants = None
combine_map_normal = None
combine_map = None
combine_map_kp = None
combine_map_des = None
model_circle_detection = None
models_dict = {}
models_initialized = False


def initialize_models():
    global models_dict
    models_dict.clear()
    for name in MODEL_NAMES:
        model_path = f"{PATH_PREFIX}/Models/{name}/model.h5"
        scaler_path = f"{PATH_PREFIX}/Models/{name}/scaler.pkl"

        model = None
        scaler = None

        if os.path.exists(model_path):
            model = keras.models.load_model(model_path, compile=False)
        else:
            print(f"WARNING: model '{name}' not found at {model_path}")

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            print(f"WARNING: scaler '{name}' not found at {scaler_path}")

        if model and scaler:
            models_dict[name] = {"model": model, "scaler": scaler}


def initialize_heavy_objects():
    global combine_map_normal, combine_map, combine_map_kp, combine_map_des
    global model_circle_detection, models_initialized
    global sift, flann, map_constants

    if models_initialized:
        return

    print("Initializing heavy models...")

    sift = cv2.SIFT_create()
    index_params = dict(algorithm=1, trees=20)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    combine_map_normal = cv2.imread(f"{PATH_PREFIX}/Map/combine_new.png", cv2.IMREAD_COLOR)
    combine_map = cv2.cvtColor(combine_map_normal, cv2.COLOR_BGR2GRAY)
    combine_map_kp, combine_map_des = sift.detectAndCompute(combine_map, None)

    model_circle_detection = YOLO(f"{PATH_PREFIX}/Models/bestFull.pt")

    with open(f"{PATH_PREFIX}/constants.json", "r") as file:
        map_constants = json.load(file)

    initialize_models()

    models_initialized = True
    print(f"Heavy models loaded ({len(models_dict)} zone models).")


def is_ready():
    return models_initialized


def predict_zone(image):
    """Run the full pipeline.

    Returns a dict with the annotated-circle geometry plus the map/zone that
    were detected, so the caller can both draw and log them.
    """
    zone_data = get_template_cordinates(image)

    map_type = get_map_type(zone_data["template_center"])
    if map_type in RETIRED_MAPS:
        raise PredictionError(
            "MAP_NOT_SUPPORTED",
            f"{RETIRED_MAPS[map_type]} is not supported. ZonePredictor currently "
            f"covers {' and '.join(SUPPORTED_MAPS.values())}.",
        )
    if map_type not in SUPPORTED_MAPS:
        raise PredictionError(
            "UNKNOWN_MAP",
            "Could not tell which map this is. Make sure the whole minimap is "
            "visible and not covered by the HUD.",
        )

    zone_value, zone_number = get_target_zone_radius_and_current_zone(
        zone_data["template_radius"], map_type
    )
    if zone_number == 0:
        raise PredictionError(
            "UNKNOWN_ZONE",
            "Could not work out which zone phase this is. The detected circle "
            "does not match any known zone size for this map.",
        )
    if zone_number >= 8:
        raise PredictionError(
            "FINAL_ZONE",
            "This is already the final zone, so there is no next zone to predict.",
        )

    model_input = np.array(
        [[zone_data["template_center"][0], zone_data["template_center"][1]]]
    )
    output = get_template_predicted_zone(map_type, zone_number, model_input)
    target_x, target_y = get_frame_predicted_zone(output, zone_data["matrix"])
    target_radius = int(
        (zone_data["frame_radius"] * zone_value) / zone_data["template_radius"]
    )

    return {
        "map_type": map_type,
        "zone_number": zone_number,
        "frame_center": zone_data["frame_center"],
        "predicted_center": (float(target_x), float(target_y)),
        "predicted_radius": target_radius,
    }


def annotate(image, prediction):
    """Draw the predicted circle and a direction arrow onto the screenshot."""
    predicted_center = np.array(prediction["predicted_center"], dtype=np.float32)
    frame_center = np.array(prediction["frame_center"], dtype=np.float32)
    target_radius = prediction["predicted_radius"]

    cv2.circle(image, tuple(map(int, predicted_center)), target_radius, (0, 255, 0), 3)

    direction = predicted_center - frame_center
    distance = np.linalg.norm(direction)
    if distance > 0:
        unit_vector = direction / distance
        arrow_end = tuple(map(int, frame_center + unit_vector * target_radius))
        cv2.arrowedLine(
            image, tuple(map(int, predicted_center)), arrow_end, (0, 0, 255), 3, tipLength=0.4
        )
    return image


def get_frame_predicted_zone(output, matrix):
    predicted_template_point = np.array([[output[0]]], dtype=np.float32)
    predicted_image_point = cv2.perspectiveTransform(
        predicted_template_point, np.linalg.inv(matrix)
    )
    target_x, target_y = predicted_image_point[0][0]
    return target_x, target_y


def get_template_predicted_zone(map_type, zone_number, model_input):
    model_name = f"{map_type}{zone_number}{zone_number + 1}"
    entry = models_dict.get(model_name)
    if entry is None:
        raise PredictionError(
            "MODEL_UNAVAILABLE",
            f"No trained model is available for this map and zone phase ({model_name}).",
            status=503,
        )
    input_data = entry["scaler"]["scaler_X"].transform(model_input).reshape((1, 1, 2))
    output_scaled = entry["model"].predict(input_data, verbose=0)
    return entry["scaler"]["scaler_y"].inverse_transform(output_scaled)


def get_template_cordinates(image):
    """Locate the zone circle and map it into template space.

    Raises PredictionError when the screenshot cannot be processed, so the
    caller never has to guess from sentinel values whether this succeeded.
    """
    zone = {
        "frame_center": (0, 0),
        "frame_radius": -1,
        "template_center": (0, 0),
        "template_radius": -1,
        "matrix": None,
    }

    results = model_circle_detection.predict(
        source=image, save=False, imgsz=1024, verbose=False
    )

    best_bbox = None
    best_confidence = 0.0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = box.conf[0]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bbox = box.xyxy[0]

    if best_bbox is None:
        raise PredictionError(
            "ZONE_NOT_DETECTED",
            "No zone circle was found in that screenshot. Make sure the white "
            "zone circle is visible on the minimap.",
        )

    processed_frame = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = sift.detectAndCompute(processed_frame, None)

    if frame_des is None or len(frame_des) < 2:
        raise PredictionError(
            "MAP_NOT_MATCHED",
            "Could not match that screenshot to a game map. Try a sharper or "
            "less cropped screenshot.",
        )

    matches = flann.knnMatch(combine_map_des, frame_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

    if len(good_matches) < 4:
        raise PredictionError(
            "MAP_NOT_MATCHED",
            "Could not match that screenshot to a game map. Try a sharper or "
            "less cropped screenshot.",
        )

    src_pts = np.float32(
        [combine_map_kp[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, _mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
    if matrix is None:
        raise PredictionError(
            "MAP_NOT_MATCHED",
            "Could not align that screenshot with a game map. Try a sharper or "
            "less cropped screenshot.",
        )

    x1, y1, x2, y2 = map(int, best_bbox)
    bbox_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)

    zone["frame_center"], zone["frame_radius"] = _centre_and_radius(bbox_pts)

    bbox_pts = cv2.perspectiveTransform(bbox_pts, matrix)
    zone["template_center"], zone["template_radius"] = _centre_and_radius(bbox_pts)
    zone["matrix"] = matrix

    # Guards the divisor used to scale the predicted radius back into frame space.
    if zone["template_radius"] <= 0:
        raise PredictionError(
            "ZONE_NOT_DETECTED",
            "The detected zone circle was too small to measure. Try a "
            "higher-resolution screenshot.",
        )

    return zone


def _centre_and_radius(pts):
    center_x = int(pts[:, 0, 0].mean())
    center_y = int(pts[:, 0, 1].mean())
    radius = int(
        max(
            abs(pts[0][0][0] - pts[1][0][0]),
            abs(pts[0][0][1] - pts[2][0][1]),
        )
        // 2
    )
    return (center_x, center_y), radius


def get_target_zone_radius_and_current_zone(radius, map_type):
    """Return (radius of the next zone, current zone number).

    (0, 0) means the radius matched no known phase for this map.
    """
    constants_key = MAP_CONSTANT_KEYS.get(map_type)
    if constants_key is None:
        return 0, 0

    tiers = map_constants[constants_key]
    for zone in range(1, 9):
        low, high = tiers[str(zone)]
        if low <= radius <= high:
            if zone == 8:
                return 0, 8
            return tiers[str(zone + 1)][0], zone
    return 0, 0


def get_map_type(center):
    """Which quadrant of the stitched template the zone centre fell into.

    Still reports the retired maps so the caller can say "Vikendi is not
    supported" rather than "unknown map".
    """
    if center[0] <= 0 or center[1] <= 0:
        return "None"
    if center[0] < QUADRANT_X and center[1] < QUADRANT_Y:
        return "e"
    if center[0] > QUADRANT_X and center[1] < QUADRANT_Y:
        return "v"
    if center[0] < QUADRANT_X and center[1] > QUADRANT_Y:
        return "s"
    if center[0] > QUADRANT_X and center[1] > QUADRANT_Y:
        return "m"
    return "Unknown"


def save_result(center_x, center_y, radius, file):
    """Debug helper: draw a circle on the full template and write it to disk."""
    result_path = f"{PATH_PREFIX}/TestResult/{file}.jpg"
    temp_map = combine_map_normal.copy()
    cv2.circle(temp_map, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)
    cv2.imwrite(result_path, temp_map)
    print(f"Results saved x : {center_x} y : {center_y} r : {radius}")
