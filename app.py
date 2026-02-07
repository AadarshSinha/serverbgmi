from flask import Flask, request, jsonify, send_file
from PIL import Image
import pickle
import keras
import cv2
import numpy as np
from ultralytics import YOLO
import traceback
import json
from pathlib import Path
import os
from flask_cors import CORS
import io

path_prefix = str(Path(__file__).parent.absolute())

app = Flask(__name__)
CORS(app)

sift = cv2.SIFT_create()
index_params = dict(algorithm=1, trees=20)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

combine_map_normal = None
combine_map = None
combine_map_kp = None
combine_map_des = None
model_circle_detection = None
models_initialized = False

with open(f"{path_prefix}/constants.json", "r") as file:
    map_constants = json.load(file)

models_dict = {}

def initialize_models():
    global models_dict
    models_dict.clear()
    model_names = ["e12", "e23", "e34", "e45", "e56", "e67", "e78",
                   "m12", "m23", "m34", "m45", "m56", "m67", "m78",
                   "v12", "v23", "v34", "v45", "v56", "v67", "v78",
                   "s12", "s23", "s34", "s45", "s56", "s67", "s78"]
    for name in model_names:
        model_path = f"{path_prefix}/Models/{name}/model.h5"
        scaler_path = f"{path_prefix}/Models/{name}/scaler.pkl"

        model = None
        scaler = None

        if os.path.exists(model_path):
            model = keras.models.load_model(model_path, compile=False)
        else:
            print(f"⚠️ Warning: Model '{name}' not found at {model_path}")

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            print(f"⚠️ Warning: Scaler '{name}' not found at {scaler_path}")

        if model and scaler:
            models_dict[name] = {"model": model, "scaler": scaler}

def initialize_heavy_objects():
    global combine_map_normal, combine_map, combine_map_kp, combine_map_des
    global model_circle_detection, models_initialized

    if models_initialized:
        return

    print("Initializing heavy models...")

    combine_map_normal = cv2.imread(f'{path_prefix}/Map/combine_new.png', cv2.IMREAD_COLOR)
    combine_map = cv2.cvtColor(combine_map_normal, cv2.COLOR_BGR2GRAY)
    combine_map_kp, combine_map_des = sift.detectAndCompute(combine_map, None)

    model_circle_detection = YOLO(f'{path_prefix}/Models/bestFull.pt')

    initialize_models()

    models_initialized = True
    print("Heavy models loaded.")

print("Server started . . . ")

@app.route("/", methods=["GET"])
def home():
    initialize_heavy_objects()
    return jsonify({"message": "Server is live!"})

@app.route("/predict", methods=["POST"])
def predict():    
    print("Got request")

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    zone_data = get_template_cordinates(image)
    print(f"zone_data = {zone_data}")
    map_type = getMapType(zone_data["template_center"])
    zone_value, zone_number = get_target_zone_radius_and_current_zone(zone_data["template_radius"], map_type)
    input = np.array([[zone_data["template_center"][0], zone_data["template_center"][1]]])
    output = get_template_predicted_zone(map_type, zone_number, input)
    target_x, target_y = get_frame_predicted_zone(output, zone_data["matrix"])
    target_radius = int((zone_data["frame_radius"] * zone_value) / zone_data["template_radius"])
    predicted_center = np.array([float(target_x), float(target_y)],dtype=np.float32)
    frame_center = np.array(zone_data["frame_center"],dtype=np.float32)
    # save_result(output[0][0], output[0][1], zone_value, "result1")
    cv2.circle(image,tuple(map(int, predicted_center)),target_radius,(0, 255, 0),3)
    direction = predicted_center - frame_center
    distance = np.linalg.norm(direction)
    if distance > 0:
        unit_vector = direction / distance
        arrow_end = frame_center + unit_vector * target_radius
        arrow_end = tuple(map(int, arrow_end))
        cv2.arrowedLine(image,tuple(map(int, predicted_center)),arrow_end,(0, 0, 255),3,tipLength=0.4)
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return {"error": "Image encoding failed"}, 500
    io_buf = io.BytesIO(buffer)
    io_buf.seek(0)
    print("Returning processed image")
    return send_file(
        io_buf,
        mimetype="image/jpeg"
    )

def get_frame_predicted_zone(output, matrix):
    predicted_template_point = np.array([[output[0]]], dtype=np.float32)
    predicted_image_point = cv2.perspectiveTransform(predicted_template_point, np.linalg.inv(matrix))  
    target_x, target_y = predicted_image_point[0][0]
    return target_x, target_y

def get_template_predicted_zone(map_type, zone_number, input):
    model_name = f'{map_type}{zone_number}{zone_number+1}'
    input_data = models_dict[model_name]["scaler"]["scaler_X"].transform(input).reshape((1, 1, 2))
    output_scaled = models_dict[model_name]["model"].predict(input_data)
    output = models_dict[model_name]["scaler"]["scaler_y"].inverse_transform(output_scaled)
    return output

def save_result(center_x, center_y, radius, file):
    result_path = f"/Users/aadarshsinha/Desktop/VsCode/serverbgmi/TestResult/{file}.jpg"
    temp_map = combine_map_normal.copy()
    cv2.circle(temp_map, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)    
    cv2.imwrite(result_path, temp_map)
    print(f"Results saved x : {center_x} y : {center_y} r : {radius}")

def get_template_cordinates(image):
    best_bbox = None
    zone = {"frame_center": (0, 0), "frame_radius": -1, "template_center": (0, 0), "template_radius": -1, "matrix": None}
    try:
        results = model_circle_detection.predict(source=image, save=False, imgsz=1024, verbose=False)
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
            raise Exception("No Zone")

        processed_frame = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = sift.detectAndCompute(processed_frame, None)

        combine_map_matches = flann.knnMatch(combine_map_des, frame_des, k=2)
        combine_map_good_matches = [m for m, n in combine_map_matches if m.distance < 0.6 * n.distance]

        if len(combine_map_good_matches) >= 4:
            src_pts = np.float32([combine_map_kp[m.queryIdx].pt for m in combine_map_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in combine_map_good_matches]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
            if matrix is not None:
                x1, y1, x2, y2 = map(int, best_bbox)
                mapped_bbox_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
                
                center_x = int(mapped_bbox_pts[:, 0, 0].mean())
                center_y = int(mapped_bbox_pts[:, 0, 1].mean())
                radius = int(max(
                    abs(mapped_bbox_pts[0][0][0] - mapped_bbox_pts[1][0][0]),
                    abs(mapped_bbox_pts[0][0][1] - mapped_bbox_pts[2][0][1])
                ) // 2) 

                zone["frame_center"] = (center_x, center_y) 
                zone["frame_radius"] =  radius

                mapped_bbox_pts = cv2.perspectiveTransform(mapped_bbox_pts, matrix)
                center_x = int(mapped_bbox_pts[:, 0, 0].mean())
                center_y = int(mapped_bbox_pts[:, 0, 1].mean())
                radius = int(max(
                    abs(mapped_bbox_pts[0][0][0] - mapped_bbox_pts[1][0][0]),
                    abs(mapped_bbox_pts[0][0][1] - mapped_bbox_pts[2][0][1])
                ) // 2)
                zone["template_center"] = (center_x, center_y) 
                zone["template_radius"] =  radius
                zone["matrix"] =  matrix
    except:
        stack_trace = traceback.format_exc()
        print(stack_trace)
    
    return zone

def get_target_zone_radius_and_current_zone(radius, map):
    match map:
        case "e":
            match radius:
                case _ if map_constants["erangle"]["1"][0] <= radius <= map_constants["erangle"]["1"][1]:
                    return map_constants["erangle"]["2"][0], 1
                case _ if map_constants["erangle"]["2"][0] <= radius <= map_constants["erangle"]["2"][1]:
                    return map_constants["erangle"]["3"][0], 2
                case _ if map_constants["erangle"]["3"][0] <= radius <= map_constants["erangle"]["3"][1]:
                    return map_constants["erangle"]["4"][0], 3
                case _ if map_constants["erangle"]["4"][0] <= radius <= map_constants["erangle"]["4"][1]:
                    return map_constants["erangle"]["5"][0], 4
                case _ if map_constants["erangle"]["5"][0] <= radius <= map_constants["erangle"]["5"][1]:
                    return map_constants["erangle"]["6"][0], 5
                case _ if map_constants["erangle"]["6"][0] <= radius <= map_constants["erangle"]["6"][1]:
                    return map_constants["erangle"]["7"][0], 6
                case _ if map_constants["erangle"]["7"][0] <= radius <= map_constants["erangle"]["7"][1]:
                    return map_constants["erangle"]["8"][0], 7
                case _ if map_constants["erangle"]["8"][0] <= radius <= map_constants["erangle"]["8"][1]:
                    return 0, 8
        case "s":
            match radius:
                case _ if map_constants["shanok"]["1"][0] <= radius <= map_constants["shanok"]["1"][1]:
                    return map_constants["shanok"]["2"][0], 1
                case _ if map_constants["shanok"]["2"][0] <= radius <= map_constants["shanok"]["2"][1]:
                    return map_constants["shanok"]["3"][0], 2
                case _ if map_constants["shanok"]["3"][0] <= radius <= map_constants["shanok"]["3"][1]:
                    return map_constants["shanok"]["4"][0], 3
                case _ if map_constants["shanok"]["4"][0] <= radius <= map_constants["shanok"]["4"][1]:
                    return map_constants["shanok"]["5"][0], 4
                case _ if map_constants["shanok"]["5"][0] <= radius <= map_constants["shanok"]["5"][1]:
                    return map_constants["shanok"]["6"][0], 5
                case _ if map_constants["shanok"]["6"][0] <= radius <= map_constants["shanok"]["6"][1]:
                    return map_constants["shanok"]["7"][0], 6
                case _ if map_constants["shanok"]["7"][0] <= radius <= map_constants["shanok"]["7"][1]:
                    return map_constants["shanok"]["8"][0], 7
                case _ if map_constants["shanok"]["8"][0] <= radius <= map_constants["shanok"]["8"][1]:
                    return 0, 8
        case "m":
            match radius:
                case _ if map_constants["miramar"]["1"][0] <= radius <= map_constants["miramar"]["1"][1]:
                    return map_constants["miramar"]["2"][0], 1
                case _ if map_constants["miramar"]["2"][0] <= radius <= map_constants["miramar"]["2"][1]:
                    return map_constants["miramar"]["3"][0], 2
                case _ if map_constants["miramar"]["3"][0] <= radius <= map_constants["miramar"]["3"][1]:
                    return map_constants["miramar"]["4"][0], 3
                case _ if map_constants["miramar"]["4"][0] <= radius <= map_constants["miramar"]["4"][1]:
                    return map_constants["miramar"]["5"][0], 4
                case _ if map_constants["miramar"]["5"][0] <= radius <= map_constants["miramar"]["5"][1]:
                    return map_constants["miramar"]["6"][0], 5
                case _ if map_constants["miramar"]["6"][0] <= radius <= map_constants["miramar"]["6"][1]:
                    return map_constants["miramar"]["7"][0], 6
                case _ if map_constants["miramar"]["7"][0] <= radius <= map_constants["miramar"]["7"][1]:
                    return map_constants["miramar"]["8"][0], 7
                case _ if map_constants["miramar"]["8"][0] <= radius <= map_constants["miramar"]["8"][1]:
                    return 0, 8
        case "v":
            match radius:
                case _ if map_constants["vikendi"]["1"][0] <= radius <= map_constants["vikendi"]["1"][1]:
                    return map_constants["vikendi"]["2"][0], 1
                case _ if map_constants["vikendi"]["2"][0] <= radius <= map_constants["vikendi"]["2"][1]:
                    return map_constants["vikendi"]["3"][0], 2
                case _ if map_constants["vikendi"]["3"][0] <= radius <= map_constants["vikendi"]["3"][1]:
                    return map_constants["vikendi"]["4"][0], 3
                case _ if map_constants["vikendi"]["4"][0] <= radius <= map_constants["vikendi"]["4"][1]:
                    return map_constants["vikendi"]["5"][0], 4
                case _ if map_constants["vikendi"]["5"][0] <= radius <= map_constants["vikendi"]["5"][1]:
                    return map_constants["vikendi"]["6"][0], 5
                case _ if map_constants["vikendi"]["6"][0] <= radius <= map_constants["vikendi"]["6"][1]:
                    return map_constants["vikendi"]["7"][0], 6
                case _ if map_constants["vikendi"]["7"][0] <= radius <= map_constants["vikendi"]["7"][1]:
                    return map_constants["vikendi"]["8"][0], 7
                case _ if map_constants["vikendi"]["8"][0] <= radius <= map_constants["vikendi"]["8"][1]:
                    return 0, 8
    return 0, 0

def getMapType(center):
    if center[0] <= 0 or center[1] <= 0:
        return "None"
    if center[0] < 1280 and center[1] < 1440:
        return "e"
    if center[0] > 1280 and center[1] < 1440:
        return "v"
    if center[0] < 1280 and center[1] > 1440:
        return "s"
    if center[0] > 1280 and center[1] > 1440:
        return "m"
    return "Unknown"

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)