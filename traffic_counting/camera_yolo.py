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

# ===== SHARED STATE BETWEEN CAMERAS =====
# This allows Camera 1 to access license plates detected by Camera 0
SHARED_VEHICLE_DATABASE = {}  # {track_id: {"plate": "ABC123", "timestamp": datetime}}
DATABASE_LOCK = threading.Lock()

def register_vehicle_plate(track_id, plate_number):
    """Camera 0 registers a vehicle's plate in shared database"""
    with DATABASE_LOCK:
        SHARED_VEHICLE_DATABASE[track_id] = {
            "plate": plate_number,
            "timestamp": datetime.datetime.now(),
            "first_seen_camera": "Camera 0",
            "assigned": False  # Track if plate has been assigned to Camera 1 vehicle
        }
        print(f"📝 Registered to database: ID {track_id} -> {plate_number}")

def get_next_available_plate():
    """
    Get the most recent unassigned plate from Camera 0
    """
    with DATABASE_LOCK:
        if not SHARED_VEHICLE_DATABASE:
            return "UNKNOWN"
        
        # Get most recent vehicle from Camera 0 (within last 30 seconds)
        recent_vehicles = []
        cutoff_time = datetime.datetime.now() - datetime.timedelta(seconds=30)
        
        for track_id, data in SHARED_VEHICLE_DATABASE.items():
            if data["timestamp"] > cutoff_time and not data.get("assigned", False):
                recent_vehicles.append((track_id, data["plate"], data["timestamp"]))
        
        if recent_vehicles:
            # Return most recent unassigned plate
            recent_vehicles.sort(key=lambda x: x[2], reverse=True)
            cam0_id, plate, _ = recent_vehicles[0]
            
            # Mark as assigned
            SHARED_VEHICLE_DATABASE[cam0_id]["assigned"] = True
            
            return plate
        
        return "UNKNOWN"


class Camera(BaseCamera):
    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)

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
        api_update_interval = 2

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

        VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

        # ===== Camera 0 Variables (OCR Detection Station) =====
        cam0_vehicle_labels = {}
        cam0_vehicle_count = 0
        cam0_ocr_queue = []  # Queue OCR requests to avoid overwhelming API

        # ===== Camera 1 Variables (Parking Lot - 3 horizontal spots) =====
        cam1_entry_count = 0
        cam1_exit_count = 0
        cam1_parking_spots = {
            1: {"occupied": False, "current_car": None, "plate": None, "history": deque(maxlen=50)},
            2: {"occupied": False, "current_car": None, "plate": None, "history": deque(maxlen=50)},
            3: {"occupied": False, "current_car": None, "plate": None, "history": deque(maxlen=50)}
        }
        cam1_already_counted_entry = deque(maxlen=50)
        cam1_already_counted_spots = {1: deque(maxlen=30), 2: deque(maxlen=30), 3: deque(maxlen=30)}
        cam1_vehicle_plates = {}  # Maps track_id to plate number
        cam1_memory = {}
        cam1_pending_entry_vehicles = deque(maxlen=10)  # Track vehicles that just entered

        while True:
            cam_id, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')
            
            if frame is None:
                print(f"⚠️ [{cam_id}] Received None frame, skipping...")
                continue

            num_frames += 1
            h, w = frame.shape[:2]

            is_camera_0 = (cam_id == 'Camera 0')
            is_camera_1 = (cam_id == 'Camera 1')

            # Run YOLO detection
            image = Image.fromarray(frame[..., ::-1])
            boxes, confidence, classes = yolo.detect_image(image)
            
            if len(boxes) == 0:
                # No detections, just yield frame
                if is_camera_0:
                    cv2.putText(frame, f"Total Vehicles: {cam0_vehicle_count}", 
                            (int(0.02 * w), int(0.05 * h)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)
                elif is_camera_1:
                    cv2.putText(frame, f"Entries: {cam1_entry_count} | Exits: {cam1_exit_count}", 
                            (int(0.02 * w), int(0.05 * h)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)
                yield cam_id, frame
                continue
            
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

            # ========== CAMERA 0: OCR DETECTION STATION ==========
            if is_camera_0:
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    bbox = track.to_tlbr()
                    track_cls = track.cls
                    track_id = track.track_id

                    if track_cls not in VEHICLE_CLASSES:
                        continue

                    midpoint = track.tlbr_midpoint(bbox)

                    # New vehicle detected - run OCR and register
                    if track_id not in cam0_vehicle_labels:
                        cam0_vehicle_count += 1
                        print(f"🚗 [Camera 0] New vehicle! ID {track_id} (Total: {cam0_vehicle_count})")
                        print(f"🔍 [Camera 0] Running OCR for ID {track_id}...")
                        
                        ocr_result = capture_and_ocr(frame, bbox)
                        cam0_vehicle_labels[track_id] = ocr_result
                        
                        # Register in shared database for Camera 1
                        register_vehicle_plate(track_id, ocr_result)
                        
                        print(f"✅ [Camera 0] ID {track_id} -> {ocr_result}")

                    # Draw bounding box
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                                (0, 255, 0), 2)
                    
                    # Display ID and plate
                    label_text = f"ID: {track_id}"
                    if track_id in cam0_vehicle_labels:
                        label_text += f" | {cam0_vehicle_labels[track_id]}"
                    
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.putText(frame, f"Type: {track_cls}", (int(bbox[0]), int(bbox[3] + 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    cv2.circle(frame, midpoint, 4, (0, 255, 0), -1)

                cv2.putText(frame, f"Total Vehicles: {cam0_vehicle_count}", 
                        (int(0.02 * w), int(0.05 * h)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 255), 2)

            # ========== CAMERA 1: PARKING LOT (3 HORIZONTAL SPOTS) ==========
            elif is_camera_1:
                memory = cam1_memory
                
                # Define entry line (vertical, left side)
                entry_line = [(int(0.35 * w), 0), (int(0.35 * w), h)]
                
                # 3 horizontal parking spots at top (evenly spaced)
                spot_y = int(0.35 * h)  # 25% down from top
                spot_lines = {
                    1: [(int(0.40 * w), spot_y), (int(0.55 * w), spot_y)],  # Left
                    2: [(int(0.60 * w), spot_y), (int(0.75 * w), spot_y)],  # Center
                    3: [(int(0.80 * w), spot_y), (int(0.95 * w), spot_y)]   # Right
                }

                # Draw entry line
                cv2.line(frame, entry_line[0], entry_line[1], (0, 0, 255), 3)
                cv2.putText(frame, "ENTRY/EXIT", (entry_line[0][0] - 50, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw parking spot lines
                for spot_id, line in spot_lines.items():
                    color = (0, 255, 0) if not cam1_parking_spots[spot_id]['occupied'] else (0, 0, 255)
                    cv2.line(frame, line[0], line[1], color, 3)
                    label_x = (line[0][0] + line[1][0]) // 2 - 40
                    label_y = line[0][1] - 10
                    cv2.putText(frame, f"SPOT {spot_id}", (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                        
                    bbox = track.to_tlbr()
                    track_cls = track.cls
                    track_id = track.track_id

                    if track_cls not in VEHICLE_CLASSES:
                        continue

                    midpoint = track.tlbr_midpoint(bbox)
                    origin_midpoint = (midpoint[0], h - midpoint[1])

                    if track_id not in memory:
                        memory[track_id] = deque(maxlen=2)

                    memory[track_id].append(midpoint)
                    
                    if len(memory[track_id]) < 2:
                        continue
                        
                    previous_midpoint = memory[track_id][0]
                    origin_previous_midpoint = (previous_midpoint[0], h - previous_midpoint[1])

                    cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

                    # Check entry/exit line
                    if Camera.intersect(midpoint, previous_midpoint, entry_line[0], entry_line[1]):
                        if track_id not in cam1_already_counted_entry:
                            angle = Camera.vector_angle(origin_midpoint, origin_previous_midpoint)
                            
                            if -90 < angle < 90:  # Entry (left to right)
                                cam1_entry_count += 1
                                cv2.line(frame, entry_line[0], entry_line[1], (0, 255, 0), 3)
                                
                                # Assign next available plate from Camera 0
                                plate = get_next_available_plate()
                                cam1_vehicle_plates[track_id] = plate
                                cam1_pending_entry_vehicles.append(track_id)
                                
                                print(f"🚗 [Camera 1] Entry: ID {track_id} | Assigned Plate: {plate}")
                            else:  # Exit (right to left)
                                cam1_exit_count += 1
                                cv2.line(frame, entry_line[0], entry_line[1], (255, 0, 0), 3)
                                print(f"🚗 [Camera 1] Exit: ID {track_id}")
                            
                            cam1_already_counted_entry.append(track_id)

                    # Check parking spots (all use downward entry logic)
                    for spot_id, line in spot_lines.items():
                        if Camera.intersect(midpoint, previous_midpoint, line[0], line[1]):
                            if track_id not in cam1_already_counted_spots[spot_id]:
                                angle = Camera.vector_angle(origin_midpoint, origin_previous_midpoint)
                                
                                # For horizontal lines at top: negative angle = downward = entry
                                is_entry = angle < 0
                                
                                if is_entry:
                                    cam1_parking_spots[spot_id]['occupied'] = True
                                    cam1_parking_spots[spot_id]['current_car'] = track_id
                                    cam1_parking_spots[spot_id]['plate'] = cam1_vehicle_plates.get(track_id, "UNKNOWN")
                                    cam1_parking_spots[spot_id]['history'].append(track_id)
                                    cv2.line(frame, line[0], line[1], (0, 255, 0), 5)
                                    
                                    mock_api_call(track_id, spot_id, cam1_vehicle_plates.get(track_id, "UNKNOWN"))
                                    print(f"🅿️  [Camera 1] Spot {spot_id}: ID {track_id} ({cam1_vehicle_plates.get(track_id, 'UNKNOWN')}) PARKED")
                                else:
                                    if cam1_parking_spots[spot_id]['current_car'] == track_id:
                                        cam1_parking_spots[spot_id]['occupied'] = False
                                        cam1_parking_spots[spot_id]['current_car'] = None
                                        cam1_parking_spots[spot_id]['plate'] = None
                                    cv2.line(frame, line[0], line[1], (255, 0, 0), 5)
                                    print(f"🅿️  [Camera 1] Spot {spot_id}: ID {track_id} LEFT")
                                
                                cam1_already_counted_spots[spot_id].append(track_id)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                                (255, 255, 255), 2)
                    
                    # Display ID and plate
                    plate = cam1_vehicle_plates.get(track_id, "UNKNOWN")
                    label_text = f"ID: {track_id} | {plate}"
                    
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.circle(frame, midpoint, 4, (255, 0, 255), -1)

                # Display counts
                cv2.putText(frame, f"Entries: {cam1_entry_count} | Exits: {cam1_exit_count}", 
                        (int(0.02 * w), int(0.05 * h)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 255), 2)

                # Display parking status (horizontal at bottom)
                status_y = int(0.92 * h)
                for spot_id in sorted(cam1_parking_spots.keys()):
                    spot_data = cam1_parking_spots[spot_id]
                    status = "OCCUPIED" if spot_data['occupied'] else "EMPTY"
                    color = (0, 0, 255) if spot_data['occupied'] else (0, 255, 0)
                    
                    # Position status below each spot
                    spot_center_x = (spot_lines[spot_id][0][0] + spot_lines[spot_id][1][0]) // 2
                    
                    cv2.putText(frame, f"S{spot_id}: {status}", (spot_center_x - 50, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if spot_data['occupied'] and spot_data['plate']:
                        cv2.putText(frame, spot_data['plate'], (spot_center_x - 40, status_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Periodic API
                all_active_tracks = set([track.track_id for track in tracker.tracks 
                                        if track.is_confirmed() and track.time_since_update <= 1])
                
                current_time = time.time()
                if current_time - last_api_update >= api_update_interval:
                    send_live_parking_status(cam1_parking_spots, cam1_vehicle_plates, 
                                            cam1_entry_count, cam1_exit_count, memory, all_active_tracks)
                    last_api_update = current_time

                if len(memory) > 50:
                    del memory[list(memory)[0]]

            yield cam_id, frame


# ======================================
# Gemini OCR Functions
# ======================================
API_KEY = "AIzaSyAIaU_xYpM2izZuisoevZLF5wxSxTKXxUw"

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
    return result["uri"], result["name"]


def delete_file(file_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={API_KEY}"
    r = requests.delete(url)


def extract_plate_ocr(image_path):
    try:
        reference_uri, reference_name = upload_file(image_path, "image/jpeg")

        gen_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}

        payload = {
            "contents": [{
                "parts": [
                    {"fileData": {"fileUri": reference_uri, "mimeType": "image/jpeg"}},
                    {"text": "Extract only the vehicle license plate alphanumeric code. No prefixes, suffixes, or labels. Just the code."}
                ]
            }]
        }

        resp = requests.post(gen_url, headers=headers, json=payload)
        resp.raise_for_status()

        result = resp.json()
        plate_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        delete_file(reference_name)
        
        return plate_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return "UNKNOWN"


def capture_and_ocr(frame, bbox):
    try:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Add padding to crop
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        
        vehicle_crop = frame[y1:y2, x1:x2]
        
        timestamp = int(time.time() * 1000)
        temp_path = f"vehicle_crop_{timestamp}.jpg"
        cv2.imwrite(temp_path, vehicle_crop)
        
        result = extract_plate_ocr(temp_path)
        
        try:
            os.remove(temp_path)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Capture Error: {e}")
        return "ERROR"


def mock_api_call(car_id, spot_id, plate):
    def send_request():
        payload = {
            "car_id": car_id,
            "spot_id": spot_id,
            "plate": plate,
            "timestamp": datetime.datetime.now().isoformat()
        }
        print(f"📡 API: {payload}")
    
    threading.Thread(target=send_request, daemon=True).start()


def send_live_parking_status(parking_spots, vehicle_plates, entry_count, exit_count, memory, all_active_tracks):
    def send_request():
        all_cars = []
        for track_id in all_active_tracks:
            car_position = {"x": None, "y": None}
            if track_id in memory and len(memory[track_id]) > 0:
                current_pos = memory[track_id][-1]
                car_position = {"x": int(current_pos[0]), "y": int(current_pos[1])}
            
            all_cars.append({
                "car_id": track_id,
                "plate": vehicle_plates.get(track_id, "UNKNOWN"),
                "coordinates": car_position
            })
        
        spots_data = []
        for spot_id, data in parking_spots.items():
            spot_info = {
                "spot_id": spot_id,
                "occupied": data['occupied'],
                "current_car": None
            }
            
            if data['current_car'] is not None:
                spot_info['current_car'] = {
                    "car_id": data['current_car'],
                    "plate": data['plate']
                }
            
            spots_data.append(spot_info)
        
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_entries": entry_count,
            "total_exits": exit_count,
            "all_active_cars": all_cars,
            "parking_spots": spots_data
        }
        
        print(f"📊 Live Status:")
        print(json.dumps(payload, indent=2))
    
    threading.Thread(target=send_request, daemon=True).start()