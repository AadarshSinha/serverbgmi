import cv2
import requests

file = "test3.jpg"
image_path = f"/Users/aadarshsinha/Desktop/VsCode/serverbgmi/TestData/{file}"

url = "http://127.0.0.1:5000/predict"
files = {"file": open(image_path, "rb")}
response = requests.post(url, files=files)
data = response.json()

image = cv2.imread(image_path)

center = tuple(map(int, data["center"]))  # Convert center to tuple of integers
radius = int(data["radius"])  # Convert radius to integer

cv2.circle(image, center, radius, (0, 0, 255), 3)

output_path = f"/Users/aadarshsinha/Desktop/VsCode/serverbgmi/TestResult/{file}"
cv2.imwrite(output_path, image)

print("Circle marked and saved successfully at:", output_path)
