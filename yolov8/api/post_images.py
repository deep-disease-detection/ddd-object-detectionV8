import requests
import json
import base64

url = "http://localhost:8000/predict/"
url2 = "https://ddd-object-detection-unyu5blyja-ew.a.run.app/predict/"


with open('yolov8/data/Papilloma_1547.jpg', 'rb') as f:
    image = f.read()
    # transform the image into a JSON string
    image_str = base64.b64encode(image).decode("utf-8")
    # Create a dictionary representing your JSON message
    message_dict = {"image": image_str}
    # Convert the dictionary to a JSON string
    message_str = json.dumps(message_dict)
    # Send the message to the API
    print('sending image base64')
    response = requests.post(url2, json=message_dict)

    print(response.json())
