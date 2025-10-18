from __future__ import division, print_function, absolute_import

import cv2
from base_camera import BaseCamera
import os
import requests
import json
import threading
import time
from queue import Queue
import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort.detection import Detection
from importlib import import_module
from collections import Counter
from collections import deque
import datetime
import math

warnings.filterwarnings('ignore')


class Camera(BaseCamera):
    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)

    # Return true if line segments AB and CD intersect
    @staticmethod
    def intersect(A, B, C, D):
        return Camera.ccw(A, C, D) != Camera.ccw(B, C, D) and Camera.ccw(A, B, C) != Camera.ccw(A, B, D)

    @staticmethod
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def vector_angle(midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    @staticmethod
    def yolo_frames(image_hub, unique_name):
        device = unique_name[1]
        show_detections = False
        last_api_update = time.time()
        api_update_interval = 2  # seconds

        gdet = import_module('tools.generate_detections')
        nn_matching = import_module('deep_sort.nn_matching')
        Tracker = import_module('deep_sort.tracker').Tracker

        max_cosine_distance = 0.3
        nn_budget = None

        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        yolo = YOLO()
        nms_max_overlap = 1.0
        num_frames = 0

        # Counters
        entry_count = 0
        exit_count = 0
        
        # Parking spot tracking - one car per spot at a time
        parking_spots = {
            1: {"occupied": False, "current_car": None, "history": deque(maxlen=50)},
            2: {"occupied": False, "current_car": None, "history": deque(maxlen=50)},
            3: {"occupied": False, "current_car": None, "history": deque(maxlen=50)},
            4: {"occupied": False, "current_car": None, "history": deque(maxlen=50)}
        }
        
        # Track which vehicles crossed which lines
        already_counted_entry = deque(maxlen=50)
        already_counted_spots = {1: deque(maxlen=30), 2: deque(maxlen=30), 
                                3: deque(maxlen=30), 4: deque(maxlen=30)}
        
        # Store OCR results per vehicle
        vehicle_labels = {}  # {track_id: "PLATE1234"}
        
        memory = {}

        while True:
            cam_id, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')
            
            if frame is None:
                break

            num_frames += 1
            h, w = frame.shape[:2]

            # Define lines - 2x2 GRID CONFIGURATION
            # Vertical entry/exit line (left side)
            entry_line = [(int(0.15 * w), 0), (int(0.15 * w), h)]
            
            # Horizontal parking spot lines in 2x2 grid
            # Top row (spots 1 and 2)
            spot_lines = {
                1: [(int(0.2 * w), int(0.30 * h)), (int(0.48 * w), int(0.30 * h))],   # Top-left
                2: [(int(0.52 * w), int(0.30 * h)), (int(0.80 * w), int(0.30 * h))],  # Top-right
                # Bottom row (spots 3 and 4)
                3: [(int(0.2 * w), int(0.70 * h)), (int(0.48 * w), int(0.70 * h))],   # Bottom-left
                4: [(int(0.52 * w), int(0.70 * h)), (int(0.80 * w), int(0.70 * h))]   # Bottom-right
            }

            image = Image.fromarray(frame[..., ::-1])
            boxes, confidence, classes = yolo.detect_image(image)
            features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                        zip(boxes, confidence, classes, features)]

            boxes_arr = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes_arr = np.array([d.cls for d in detections])
            indices = preprocessing.non_max_suppression(boxes_arr, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)

            # Draw entry/exit line (RED vertical)
            cv2.line(frame, entry_line[0], entry_line[1], (0, 0, 255), 3)
            cv2.putText(frame, "ENTRY/EXIT", (entry_line[0][0] - 50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw parking spot lines (RED horizontal) in 2x2 grid
            for spot_id, line in spot_lines.items():
                cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
                # Position labels above each line
                label_x = (line[0][0] + line[1][0]) // 2 - 40
                label_y = line[0][1] - 10
                cv2.putText(frame, f"SPOT {spot_id}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                    
                bbox = track.to_tlbr()
                track_cls = track.cls
                track_id = track.track_id

                midpoint = track.tlbr_midpoint(bbox)
                origin_midpoint = (midpoint[0], h - midpoint[1])

                if track_id not in memory:
                    memory[track_id] = deque(maxlen=2)

                memory[track_id].append(midpoint)
                
                if len(memory[track_id]) < 2:
                    continue
                    
                previous_midpoint = memory[track_id][0]
                origin_previous_midpoint = (previous_midpoint[0], h - previous_midpoint[1])

                # Draw tracking line
                cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

                # Check entry/exit line crossing
                if Camera.intersect(midpoint, previous_midpoint, entry_line[0], entry_line[1]):
                    if track_id not in already_counted_entry:
                        angle = Camera.vector_angle(origin_midpoint, origin_previous_midpoint)
                        
                        # Determine direction (left to right = entry, right to left = exit)
                        if -90 < angle < 90:  # Moving right (entry)
                            entry_count += 1
                            cv2.line(frame, entry_line[0], entry_line[1], (0, 255, 0), 3)
                            
                            # Get OCR label for new entry using Gemini
                            if track_id not in vehicle_labels:
                                print(f"🔍 Running OCR for vehicle ID {track_id}...")
                                ocr_result = call_ocr_api(frame, bbox)
                                vehicle_labels[track_id] = ocr_result
                                print(f"🚗 New Entry: ID {track_id} -> {ocr_result}")
                        else:  # Moving left (exit)
                            exit_count += 1
                            cv2.line(frame, entry_line[0], entry_line[1], (255, 0, 0), 3)
                            print(f"🚗 Exit: ID {track_id}")
                        
                        already_counted_entry.append(track_id)

                # Check parking spot line crossings
                for spot_id, line in spot_lines.items():
                    if Camera.intersect(midpoint, previous_midpoint, line[0], line[1]):
                        if track_id not in already_counted_spots[spot_id]:
                            angle = Camera.vector_angle(origin_midpoint, origin_previous_midpoint)
                            
                            # Determine if it's entry or exit based on spot and direction
                            is_entry = False
                            if spot_id in [1, 2]:  # Spots 1 and 2: up = entry, down = exit
                                is_entry = angle > 0
                            else:  # Spots 3 and 4: down = entry, up = exit
                                is_entry = angle < 0
                            
                            if is_entry:
                                # Car entering parking spot
                                parking_spots[spot_id]['occupied'] = True
                                parking_spots[spot_id]['current_car'] = track_id
                                parking_spots[spot_id]['history'].append(track_id)
                                
                                # Flash the line green for entry
                                cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
                                
                                # Mock API call
                                mock_api_call(track_id, spot_id)
                                print(f"🅿️  Spot {spot_id}: Car {track_id} ENTERED (parked)")
                            else:
                                # Car exiting parking spot
                                if parking_spots[spot_id]['current_car'] == track_id:
                                    parking_spots[spot_id]['occupied'] = False
                                    parking_spots[spot_id]['current_car'] = None
                                
                                # Flash the line blue for exit
                                cv2.line(frame, line[0], line[1], (255, 0, 0), 3)
                                
                                print(f"🅿️  Spot {spot_id}: Car {track_id} EXITED (left)")
                            
                            already_counted_spots[spot_id].append(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                            (255, 255, 255), 2)
                
                # Display ID and OCR label
                label_text = f"ID: {track_id}"
                if track_id in vehicle_labels:
                    label_text += f" | {vehicle_labels[track_id]}"
                
                cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display X,Y coordinates
                coord_text = f"X:{midpoint[0]} Y:{midpoint[1]}"
                cv2.putText(frame, coord_text, (int(bbox[0]), int(bbox[3] + 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Draw center point marker
                cv2.circle(frame, midpoint, 4, (255, 0, 255), -1)

            # Display entry/exit counts
            cv2.putText(frame, f"Entries: {entry_count} | Exits: {exit_count}", 
                    (int(0.02 * w), int(0.05 * h)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)

            # Display parking spot information
            y_offset = int(0.05 * h)
            for spot_id in sorted(parking_spots.keys()):
                spot_data = parking_spots[spot_id]
                status = "🚗 OCCUPIED" if spot_data['occupied'] else "✅ EMPTY"
                info_text = f"Spot {spot_id}: {status}"
                
                cv2.putText(frame, info_text, (int(0.65 * w), y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 0, 255) if spot_data['occupied'] else (0, 255, 0), 2)
                
                # Show current car if occupied
                if spot_data['occupied'] and spot_data['current_car']:
                    car_id = spot_data['current_car']
                    label = vehicle_labels.get(car_id, "UNKNOWN")
                    car_text = f"  Car: {car_id} ({label})"
                    cv2.putText(frame, car_text, (int(0.65 * w), y_offset + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 60
                else:
                    y_offset += 35

            # Collect all currently active track IDs
            all_active_tracks = set([track.track_id for track in tracker.tracks 
                                    if track.is_confirmed() and track.time_since_update <= 1])
            
            # Periodic live parking status API call
            current_time = time.time()
            if current_time - last_api_update >= api_update_interval:
                send_live_parking_status(parking_spots, vehicle_labels, entry_count, exit_count, 
                                        memory, all_active_tracks)
                last_api_update = current_time

            # Clean up old memory
            if len(memory) > 50:
                del memory[list(memory)[0]]

            yield cam_id, frame
# Gemini API Configuration
API_KEY = "AIzaSyAIaU_xYpM2izZuisoevZLF5wxSxTKXxUw"  # 🔑 Add your Gemini API key

# ======================================
# Gemini OCR Functions (DO NOT MODIFY)
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


def delete_file(file_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={API_KEY}"
    r = requests.delete(url)
    if r.status_code == 200:
        print(f"🗑️ Deleted {file_name}")
    else:
        print(f"⚠️ Failed to delete {file_name}: {r.status_code} - {r.text}")


def extract_plate_ocr(image_path):
    """Extract license plate using Gemini API"""
    try:
        # Upload to Gemini
        reference_uri, reference_name = upload_file(image_path, "image/jpeg")

        # Call Gemini API
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
        plate_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Cleanup uploaded file
        delete_file(reference_name)
        
        return plate_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return "UNKNOWN"


def call_ocr_api(frame, bbox):
    """Wrapper to extract OCR from vehicle crop"""
    try:
        
        # Save temporary image
        temp_path = "testpic.png"
        
        # Call Gemini OCR
        result = extract_plate_ocr(temp_path)
        
            
        return result
    except Exception as e:
        print(f"OCR Wrapper Error: {e}")
        return "ERROR"


def mock_api_call(car_id, spot_id):
    """Simulate API call for parking spot occupation"""
    def send_request():
        payload = {
            "car_id": car_id,
            "spot_id": spot_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print(f"📡 API Call: {payload}")
        # Uncomment below to make actual API call
        # try:
        #     response = requests.post("YOUR_API_ENDPOINT", json=payload, timeout=5)
        #     print(f"API Response: {response.status_code}")
        # except Exception as e:
        #     print(f"API Error: {e}")
    
    threading.Thread(target=send_request, daemon=True).start()

def send_live_parking_status(parking_spots, vehicle_labels, entry_count, exit_count, memory, all_active_tracks):
    """Send periodic live parking status to API"""
    def send_request():
        # Prepare data for ALL currently tracked cars
        all_cars = []
        for track_id in all_active_tracks:
            # Get current coordinates from memory
            car_position = {"x": None, "y": None}
            if track_id in memory and len(memory[track_id]) > 0:
                current_pos = memory[track_id][-1]  # Get most recent position
                car_position = {"x": int(current_pos[0]), "y": int(current_pos[1])}
            
            all_cars.append({
                "car_id": track_id,
                "label": vehicle_labels.get(track_id, "UNKNOWN"),
                "coordinates": car_position
            })
        
        # Prepare parking spots data
        spots_data = []
        for spot_id, data in parking_spots.items():
            spot_info = {
                "spot_id": spot_id,
                "occupied": data['occupied'],
                "current_car": None,
                "history": list(data['history'])  # Convert deque to list for JSON
            }
            
            # If occupied, include current car details
            if data['current_car'] is not None:
                car_id = data['current_car']
                car_position = {"x": None, "y": None}
                if car_id in memory and len(memory[car_id]) > 0:
                    current_pos = memory[car_id][-1]
                    car_position = {"x": int(current_pos[0]), "y": int(current_pos[1])}
                
                spot_info['current_car'] = {
                    "car_id": car_id,
                    "label": vehicle_labels.get(car_id, "UNKNOWN"),
                    "coordinates": car_position
                }
            
            spots_data.append(spot_info)
        
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_entries": entry_count,
            "total_exits": exit_count,
            "all_active_cars": all_cars,
            "parking_spots": spots_data
        }
        
        print(f"📊 Live Status API Call:")
        print(json.dumps(payload, indent=2))
        
        # Uncomment below to make actual API call
        # try:
        #     response = requests.post("YOUR_LIVE_STATUS_ENDPOINT", json=payload, timeout=5)
        #     print(f"Live Status API Response: {response.status_code}")
        # except Exception as e:
        #     print(f"Live Status API Error: {e}")
    
    # Run API call in separate thread
    threading.Thread(target=send_request, daemon=True).start()