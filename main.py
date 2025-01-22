import requests
import cv2
from dotenv import load_dotenv
import os
import threading

# Load .env file
load_dotenv()

# Azure Custom Vision Configuration
PREDICTION_KEY = os.getenv("PREDICTION_KEY")  # Your Prediction Key from .env
ENDPOINT = os.getenv("ENDPOINT")  # Your Endpoint from .env

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream",
}

def detect_image(image):
    """Send the image to the Azure Custom Vision API for prediction."""
    response = requests.post(
        f"{ENDPOINT}",
        headers=headers,
        data=image
    )
    if response.status_code == 200:
        return response.json().get("predictions", [])
    else:
        print("Error:", response.status_code, response.text)
        return []

def process_predictions(predictions):
    """Extract and display the most likely tag and its probability."""
    for prediction in predictions:
        print(f"Tag: {prediction['tagName']}, Probability: {prediction['probability']:.2f}")
    return predictions[0] if predictions else None

def async_detect_image(image, callback):
    """Perform asynchronous prediction."""
    def task():
        predictions = detect_image(image)
        callback(predictions)
    thread = threading.Thread(target=task)
    thread.start()

def prediction_callback(predictions):
    """Handle predictions returned from the async_detect_image."""
    if predictions:
        top_prediction = process_predictions(predictions)
        if top_prediction:
            tag = top_prediction['tagName']
            probability = top_prediction['probability']
            print(f"Detected: {tag} ({probability:.2f})")

def main():
    # Start OpenCV Video Capture
    cap = cv2.VideoCapture(0)

    # Optimize frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            _, image_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            async_detect_image(image_encoded.tobytes(), prediction_callback)

        frame_count += 1
        cv2.imshow("Real-Time Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
