import requests
import cv2
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Azure Custom Vision Configuration
PREDICTION_KEY = os.getenv("PREDICTION_KEY")
ENDPOINT = os.getenv("ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
ITERATION_NAME = os.getenv("ITERATION_NAME")

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream",
}


def detect_image(image):
    """Send the image to the Azure Custom Vision API for prediction."""
    response = requests.post(
        f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/image",
        headers=headers,
        data=image
    )

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        return predictions
    else:
        print("Error:", response.status_code, response.text)
        return []


def process_predictions(predictions):
    """Extract the most likely tag and its probability."""
    for prediction in predictions:
        print(f"Tag: {prediction['tagName']}, Probability: {prediction['probability']:.2f}")
    return predictions[0] if predictions else None


def main():
    # Start OpenCV Video Capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the real-time feed
        cv2.imshow("Real-Time Feed", frame)

        # Send frame to Azure Custom Vision for prediction
        _, image_encoded = cv2.imencode('.jpg', frame)
        predictions = detect_image(image_encoded.tobytes())

        # Process and display predictions
        if predictions:
            top_prediction = process_predictions(predictions)
            if top_prediction:
                tag = top_prediction['tagName']
                probability = top_prediction['probability']
                print(f"Detected: {tag} ({probability:.2f})")

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
