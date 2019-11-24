import json
import cv2
import requests

print("Hi")

server_addr = 'http://143.248.49.66:5009'
url = server_addr + '/checkObject'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
img = cv2.imread('heungmin.jpeg')
_, img_encoded = cv2.imencode('.jpg', img)

# send http request with image and receive response
img_encoded = img_encoded.tostring()

response = requests.post(url, data=img_encoded, headers=headers)
print(response.text)
