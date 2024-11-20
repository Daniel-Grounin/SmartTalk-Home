import requests
from ultralytics import YOLO
import cv2
import time
import random
import ollama
from gtts import gTTS
import os
import re

model = YOLO('yolov8n.pt')

def get_simulated_sensor_data():
    soil_moisture = random.randint(15, 80)
    light_level = random.randint(100, 500)
    return soil_moisture, light_level

def detect_person_yolo():
    cap = cv2.VideoCapture(0)
    person_detected = False

    print("Checking for person using YOLOv8...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:
                    xyxy = box.xyxy.tolist()
                    x1, y1, x2, y2 = xyxy[0]
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Person Detected", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    person_detected = True

        cv2.imshow("Detection", frame)
        if person_detected:
            break

    cap.release()
    cv2.destroyAllWindows()
    return person_detected


def chat_with_tinydolphin(user_input):
    try:
        stream = ollama.chat(
            model="tinydolphin",
            messages=[{'role': 'user', 'content': f"Dolphin you are the plant: {user_input}"}],
            stream=True
        )

        response = ""
        for chunk in stream:
            response += chunk['message']['content']
        return response

    except Exception as e:
        return f"An error occurred: {e}"

def smart_plant_conversation():
    print("Starting Talking Smart Plant Simulation...")
    while True:
        # Simulate soil moisture and light level
        soil_moisture, light_level = get_simulated_sensor_data()
        print(f"Simulated Soil Moisture: {soil_moisture}%, Light Level: {light_level} lux")

        person_detected = detect_person_yolo()
        print("Checking for person...")

        if person_detected:
            print("Person detected!")
            if soil_moisture < 30:  # If plant is dry
                response = chat_with_tinydolphin(
                    "You are thirsty. Do you need water?")
            else:
                response = chat_with_tinydolphin("You are well hydrated Plant")

            print(f"Plant says: {response}")

            tts = gTTS(text=response, lang='en')
            tts.save("response_english.mp3")
            os.system("start response_english.mp3")

        else:
            print("No person detected.")

        time.sleep(5)


if __name__ == "__main__":
    smart_plant_conversation()
