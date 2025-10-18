import cv2
import time
import requests
import json

API_KEY = "AIzaSyAIaU_xYpM2izZuisoevZLF5wxSxTKXxUw"  # 🔑 your Gemini API key



# ======================================
# 1️⃣ Upload Function
# ======================================
def upload_file(file_path, mime_type):
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart&key={API_KEY}"
    metadata = {"file": {"displayName": file_path.split('/')[-1]}}

    with open(file_path, "rb") as f:
        data = (
            b"--boundary\r\n"
            b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
            + json.dumps(metadata).encode("utf-8")
            + b"\r\n--boundary\r\n"
            + f"Content-Type: ".encode("utf-8") + mime_type.encode("utf-8") + b"\r\n\r\n"
            + f.read()
            + b"\r\n--boundary--"
        )

    headers = {"Content-Type": "multipart/related; boundary=boundary"}
    r = requests.post(url, headers=headers, data=data)
    r.raise_for_status()
    result = r.json()["file"]
    print(f"✅ Uploaded: {result['displayName']}")
    return result["uri"], result["name"]


# ======================================
# 2️⃣ Delete Function
# ======================================
def delete_file(file_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={API_KEY}"
    r = requests.delete(url)
    if r.status_code == 200:
        print(f"🗑️ Deleted {file_name}")
    else:
        print(f"⚠️ Failed to delete {file_name}: {r.status_code} - {r.text}")


# ======================================
# 4️⃣ Main Logic: Compare Captured vs Local Image
# ======================================
def main():

    # Local reference image 
    reference_image = "Untitled.png"

    # Upload both to Gemini
    reference_uri, reference_name = upload_file(reference_image, "image/jpeg")

    # Compare using Gemini
    gen_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "fileData": {"fileUri": reference_uri, "mimeType": "image/jpeg"}
                    },
                    {
                        "text": (
                            "From the image, extract only the unique, primary alphanumeric code of the vehicle's license plate. "
                            "Do not include any surrounding words, labels, prefixes, or suffixes such as 'REGISTERED', "
                            "'NCR MV FILE NO.', or hyphens. Respond with nothing but the exact code."
                        )
                    }
                ]
            }
        ]
    }

    resp = requests.post(gen_url, headers=headers, json=payload)
    resp.raise_for_status()

    result = resp.json()
    print(result["candidates"][0]["content"]["parts"][0]["text"])

    # Cleanup both uploaded files
    delete_file(reference_name)


if __name__ == "__main__":
    main()