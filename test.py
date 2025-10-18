import easyocr
import cv2

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # 'en' for English

# Load your image
image_path = "Untitled.png"

# Run OCR
results = reader.readtext(image_path)

# Display results
print("Detected text:")
for (bbox, text, prob) in results:
    print(f"{text} (confidence: {prob:.2f})")

# Optional: Draw boxes on image
img = cv2.imread(image_path)
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(img, text, (top_left[0], top_left[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Save output image with detected text boxes
cv2.imwrite("output_with_text.jpg", img)
print("Output saved as output_with_text.jpg")
