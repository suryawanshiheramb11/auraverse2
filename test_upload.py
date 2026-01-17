import requests

url = "http://127.0.0.1:8000/scan"
file_path = "/Users/suryawanshiheramb11/Downloads/training data/Celeb-synthesis/id31_id1_0007.mp4"

with open(file_path, "rb") as f:
    files = {"file": ("test_fake.mp4", f, "video/mp4")}
    print(f"Uploading: {file_path}")
    response = requests.post(url, files=files)
    
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
