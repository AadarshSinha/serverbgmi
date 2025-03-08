from flask import Flask, request, jsonify
from PIL import Image
import pickle
import keras
import cv2
import numpy as np
from ultralytics import YOLO
import traceback
import json
from pathlib import Path


def getPath():
    script_dir = str(Path(__file__).parent)
    prefix = script_dir.split("serverbgmi")[0] + "serverbgmi"
    return prefix
path_prefix = getPath()
print(f"Path prefix : {path_prefix}")
app = Flask(__name__)
sift = cv2.SIFT_create()
index_params = dict(algorithm=1, trees=20)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
combine_map_normal = cv2.imread(f'{path_prefix}/Map/combine_new.png',  cv2.IMREAD_COLOR)
combine_map = cv2.cvtColor(combine_map_normal, cv2.COLOR_BGR2GRAY)
combine_map_kp, combine_map_des = sift.detectAndCompute(combine_map, None)
with open(f"{path_prefix}/constants.json", "r") as file:
    map_constants = json.load(file)
model_circle_detection = YOLO(f'{path_prefix}/Models/bestFull.pt')
model_path = f"{path_prefix}/Models/e12/model.h5"
scaler_path = f"{path_prefix}/Models/e12/scaler.pkl"
model_e12 = keras.models.load_model(model_path, compile=False)

with open(scaler_path, 'rb') as f:
    scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    print("Got request")
    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)    
    zone_data = get_template_cordinates(image)
    print(f" ZONE DATA : {zone_data}")
    input = np.array([[zone_data["template_center"][0], zone_data["template_center"][1]]])
    input_data = scaler_X.transform(input).reshape((1, 1, 2))
    output_scaled = model_e12.predict(input_data)
    output = scaler_y.inverse_transform(output_scaled)

    print(f"OUTPUT : {output}")
    target_x = (zone_data["frame_center"][0]*output[0][0])/zone_data["template_center"][0]
    target_y = (zone_data["frame_center"][1]*output[0][1])/zone_data["template_center"][1]
    target_radius = (zone_data["frame_radius"] * get_zone_value(zone_data["template_radius"], "Erangle"))/zone_data["template_radius"]
    target_zone = {"center": (target_x, target_y), "radius": target_radius}
    print(f"Predicted zone cordinated on frame {target_zone}")

    return target_zone

def get_template_cordinates(image):
    best_bbox = None
    zone = {"frame_center": (0, 0), "frame_radius": -1, "template_center": (0, 0), "template_radius": -1}
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
    except:
        stack_trace = traceback.format_exc()
        print(stack_trace)
    
    return zone

def get_zone_value(radius, map):
    match map:
        case "Erangle":
            match radius:
                case _ if map_constants["erangle"]["1"][0] <= radius <= map_constants["erangle"]["1"][1]:
                    return map_constants["erangle"]["2"][0]
                case _ if map_constants["erangle"]["2"][0] <= radius <= map_constants["erangle"]["2"][1]:
                    return map_constants["erangle"]["3"][0]
                case _ if map_constants["erangle"]["3"][0] <= radius <= map_constants["erangle"]["3"][1]:
                    return map_constants["erangle"]["4"][0]
                case _ if map_constants["erangle"]["4"][0] <= radius <= map_constants["erangle"]["4"][1]:
                    return map_constants["erangle"]["5"][0]
                case _ if map_constants["erangle"]["5"][0] <= radius <= map_constants["erangle"]["5"][1]:
                    return map_constants["erangle"]["6"][0]
                case _ if map_constants["erangle"]["6"][0] <= radius <= map_constants["erangle"]["6"][1]:
                    return map_constants["erangle"]["7"][0]
                case _ if map_constants["erangle"]["7"][0] <= radius <= map_constants["erangle"]["7"][1]:
                    return map_constants["erangle"]["8"][0]
                case _ if map_constants["erangle"]["8"][0] <= radius <= map_constants["erangle"]["8"][1]:
                    return 0
        case "Shanok":
            match radius:
                case _ if map_constants["shanok"]["1"][0] <= radius <= map_constants["shanok"]["1"][1]:
                    return map_constants["shanok"]["2"][0]
                case _ if map_constants["shanok"]["2"][0] <= radius <= map_constants["shanok"]["2"][1]:
                    return map_constants["shanok"]["3"][0]
                case _ if map_constants["shanok"]["3"][0] <= radius <= map_constants["shanok"]["3"][1]:
                    return map_constants["shanok"]["4"][0]
                case _ if map_constants["shanok"]["4"][0] <= radius <= map_constants["shanok"]["4"][1]:
                    return map_constants["shanok"]["5"][0]
                case _ if map_constants["shanok"]["5"][0] <= radius <= map_constants["shanok"]["5"][1]:
                    return map_constants["shanok"]["6"][0]
                case _ if map_constants["shanok"]["6"][0] <= radius <= map_constants["shanok"]["6"][1]:
                    return map_constants["shanok"]["7"][0]
                case _ if map_constants["shanok"]["7"][0] <= radius <= map_constants["shanok"]["7"][1]:
                    return map_constants["shanok"]["8"][0]
                case _ if map_constants["shanok"]["8"][0] <= radius <= map_constants["shanok"]["8"][1]:
                    return 0
        case "Miramar":
            match radius:
                case _ if map_constants["miramar"]["1"][0] <= radius <= map_constants["miramar"]["1"][1]:
                    return map_constants["miramar"]["2"][0]
                case _ if map_constants["miramar"]["2"][0] <= radius <= map_constants["miramar"]["2"][1]:
                    return map_constants["miramar"]["3"][0]
                case _ if map_constants["miramar"]["3"][0] <= radius <= map_constants["miramar"]["3"][1]:
                    return map_constants["miramar"]["4"][0]
                case _ if map_constants["miramar"]["4"][0] <= radius <= map_constants["miramar"]["4"][1]:
                    return map_constants["miramar"]["5"][0]
                case _ if map_constants["miramar"]["5"][0] <= radius <= map_constants["miramar"]["5"][1]:
                    return map_constants["miramar"]["6"][0]
                case _ if map_constants["miramar"]["6"][0] <= radius <= map_constants["miramar"]["6"][1]:
                    return map_constants["miramar"]["7"][0]
                case _ if map_constants["miramar"]["7"][0] <= radius <= map_constants["miramar"]["7"][1]:
                    return map_constants["miramar"]["8"][0]
                case _ if map_constants["miramar"]["8"][0] <= radius <= map_constants["miramar"]["8"][1]:
                    return 0
        case "Vikendi":
            match radius:
                case _ if map_constants["vikendi"]["1"][0] <= radius <= map_constants["vikendi"]["1"][1]:
                    return map_constants["vikendi"]["2"][0]
                case _ if map_constants["vikendi"]["2"][0] <= radius <= map_constants["vikendi"]["2"][1]:
                    return map_constants["vikendi"]["3"][0]
                case _ if map_constants["vikendi"]["3"][0] <= radius <= map_constants["vikendi"]["3"][1]:
                    return map_constants["vikendi"]["4"][0]
                case _ if map_constants["vikendi"]["4"][0] <= radius <= map_constants["vikendi"]["4"][1]:
                    return map_constants["vikendi"]["5"][0]
                case _ if map_constants["vikendi"]["5"][0] <= radius <= map_constants["vikendi"]["5"][1]:
                    return map_constants["vikendi"]["6"][0]
                case _ if map_constants["vikendi"]["6"][0] <= radius <= map_constants["vikendi"]["6"][1]:
                    return map_constants["vikendi"]["7"][0]
                case _ if map_constants["vikendi"]["7"][0] <= radius <= map_constants["vikendi"]["7"][1]:
                    return map_constants["vikendi"]["8"][0]
                case _ if map_constants["vikendi"]["8"][0] <= radius <= map_constants["vikendi"]["8"][1]:
                    return 0
    return 0

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
