import cv2
import numpy as np
import time
import threading
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import deque
from ultralytics import YOLO

STREAM_PORT = 5000
YOLO_EVERY_N = 3
INPUT_WIDTH = 480
CONF_THRESH = 0.38
THERMAL_DURATION = 12
BAG_DELAY = 2
CENTROID_DIST = 90
PERSON_CLASS = 0
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}
ZONES = [("Entry Hall", 0.0, 0.2), ("Left Side", 0.2, 0.4), ("Center Lobby", 0.4, 0.6), ("Right Side", 0.6, 0.8), ("Exit Path", 0.8, 1.0)]
state = {"status": "CLEAR", "fps": 0, "persons": [], "face_active": False, "bag": None, "thermal_active": False, "weapon_conf": 0, "log": [], "frame_count": 0}
state_lock = threading.Lock()
latest_jpg = None
latest_jpg_lock = threading.Lock()
weapon_jpg = None
weapon_jpg_lock = threading.Lock()
raw_frame = None
raw_frame_lock = threading.Lock()

def log_event(msg, level="info"):
    ts = time.strftime("%H:%M:%S")
    print("[" + ts + "] " + msg)
    entry = {"ts": ts, "msg": msg, "level": level}
    with state_lock:
        state["log"].insert(0, entry)
        if len(state["log"]) > 50:
            state["log"].pop()

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera failed to open")
        self.frame = frame
        self.lock = threading.Lock()
        self.running = True
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()
        log_event("Camera connected")
    def loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
    def read(self):
        with self.lock:
            return self.frame.copy()
    def stop(self):
        self.running = False
        self.cap.release()

class Tracker:
    def __init__(self):
        self.next_id = 1
        self.people = {}
    def get_zone(self, cx, frame_w):
        ratio = cx / max(frame_w, 1)
        for name, lo, hi in ZONES:
            if lo <= ratio < hi:
                return name
        return "Unknown"
    def update(self, boxes, frame_w):
        now = time.time()
        centroids = []
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroids.append((cx, cy, x1, y1, x2, y2))
        matched = set()
        used = set()
        for pid in self.people:
            p = self.people[pid]
            best_dist = CENTROID_DIST
            best_i = -1
            for i in range(len(centroids)):
                if i in used:
                    continue
                cx, cy = centroids[i][0], centroids[i][1]
                d = math.hypot(cx - p["cx"], cy - p["cy"])
                if d < best_dist:
                    best_dist = d
                    best_i = i
            if best_i >= 0:
                matched.add(pid)
                used.add(best_i)
                cx, cy, x1, y1, x2, y2 = centroids[best_i]
                zone = self.get_zone(cx, frame_w)
                if p["zone"] != zone:
                    if not p["path"] or p["path"][-1] != zone:
                        p["path"].append(zone)
                    p["path"] = p["path"][-6:]
                p["cx"] = cx
                p["cy"] = cy
                p["box"] = (x1, y1, x2, y2)
                p["seen"] = now
                p["zone"] = zone
                p["in_frame"] = True
        for i in range(len(centroids)):
            if i in used:
                continue
            cx, cy, x1, y1, x2, y2 = centroids[i]
            zone = self.get_zone(cx, frame_w)
            self.people[self.next_id] = {"cx": cx, "cy": cy, "box": (x1, y1, x2, y2), "seen": now, "first": time.strftime("%H:%M:%S"), "zone": zone, "path": [zone], "in_frame": True, "color_top": "-", "color_bot": "-"}
            log_event("Person " + str(self.next_id) + " detected in " + zone)
            self.next_id += 1
        for pid in self.people:
            if pid not in matched and self.people[pid]["in_frame"]:
                self.people[pid]["in_frame"] = False
                log_event("Person " + str(pid) + " left frame", "warn")
        to_remove = []
        for pid in self.people:
            p = self.people[pid]
            if not p["in_frame"] and now - p["seen"] > 20:
                to_remove.append(pid)
        for pid in to_remove:
            del self.people[pid]
        return self.people

COLOR_RANGES = {"Red": ([0, 100, 70], [10, 255, 255]), "Orange": ([11, 100, 70], [25, 255, 255]), "Yellow": ([26, 100, 70], [34, 255, 255]), "Green": ([35, 50, 40], [85, 255, 255]), "Blue": ([86, 50, 50], [130, 255, 255]), "Purple": ([131, 40, 40], [160, 255, 255]), "Pink": ([161, 40, 80], [175, 255, 255]), "White": ([0, 0, 190], [180, 25, 255]), "Black": ([0, 0, 0], [180, 255, 45]), "Gray": ([0, 0, 46], [180, 25, 189]), "Brown": ([10, 50, 20], [20, 200, 150])}

def find_color(region):
    if region is None or region.size == 0:
        return "-"
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    best_name = "Unknown"
    best_count = 0
    for name in COLOR_RANGES:
        lo, hi = COLOR_RANGES[name]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_name = name
    return best_name

def get_clothing_colors(frame, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    top_y1 = y1 + int(h * 0.15)
    top_y2 = y1 + int(h * 0.45)
    bot_y1 = y1 + int(h * 0.55)
    bot_y2 = y1 + int(h * 0.85)
    top_region = frame[top_y1:top_y2, x1:x2]
    bot_region = frame[bot_y1:bot_y2, x1:x2]
    return find_color(top_region), find_color(bot_region)

def load_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(path)
    if det.empty():
        raise RuntimeError("Could not load face detector")
    log_event("Identity protection loaded")
    return det

def blur_faces(frame, detector, person_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = False
    h_frame, w_frame = frame.shape[:2]
    for x1, y1, x2, y2 in person_boxes:
        head_y1 = max(0, y1)
        head_y2 = min(h_frame, y1 + int((y2 - y1) * 0.52))
        head_x1 = max(0, x1)
        head_x2 = min(w_frame, x2)
        if head_y2 <= head_y1 or head_x2 <= head_x1:
            continue
        head_roi = gray[head_y1:head_y2, head_x1:head_x2]
        faces = detector.detectMultiScale(head_roi, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (26, 26))
        for fx, fy, fw, fh in faces:
            gx1 = head_x1 + fx
            gy1 = head_y1 + fy
            gx2 = gx1 + fw
            gy2 = gy1 + fh
            face_roi = frame[gy1:gy2, gx1:gx2]
            if face_roi.size == 0:
                continue
            frame[gy1:gy2, gx1:gx2] = cv2.GaussianBlur(face_roi, (51, 51), 30)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 210, 255), 1)
            cv2.putText(frame, "IDENTITY PROTECTED", (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 210, 255), 1)
            found = True
    return frame, found

GUN_POLY = np.array([[0.10, 0.45], [0.15, 0.40], [0.55, 0.38], [0.62, 0.32], [0.78, 0.32], [0.90, 0.36], [0.90, 0.44], [0.78, 0.48], [0.62, 0.48], [0.60, 0.55], [0.62, 0.68], [0.55, 0.70], [0.50, 0.55], [0.15, 0.55], [0.10, 0.45]], dtype=np.float32)

def draw_weapon_shape(img, x1, y1, x2, y2, t):
    w = x2 - x1
    h = y2 - y1
    pts = (GUN_POLY * np.array([[w, h]]) + np.array([[x1, y1]])).astype(np.int32)
    layers = [(6, (0, 0, 180), 0.35), (3, (0, 80, 255), 0.55), (1, (0, 160, 255), 0.75), (0, (0, 220, 255), 1.0)]
    for expand, color, alpha in layers:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    pulse = int(abs(math.sin(t * 5)) * 120) + 135
    cv2.polylines(img, [pts], True, (0, 0, pulse), 2)
    return img

def apply_thermal(frame, bag_box):
    global weapon_jpg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    h, w = thermal.shape[:2]
    t = time.time()
    scan_y = int((t * 55) % h)
    cv2.line(thermal, (0, scan_y), (w, scan_y), (255, 255, 200), 1)
    weapon_crop_data = None
    if bag_box:
        bx1, by1, bx2, by2 = bag_box
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        by2 = min(h, by2)
        if bx2 > bx1 and by2 > by1:
            cv2.rectangle(thermal, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(thermal, "BAG 1", (bx1, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
            margin = 12
            wx1 = bx1 + margin
            wy1 = by1 + margin
            wx2 = bx2 - margin
            wy2 = by2 - margin
            if wx2 > wx1 and wy2 > wy1:
                heat = thermal[wy1:wy2, wx1:wx2]
                if heat.size > 0:
                    bright = np.clip(heat.astype(np.int32) + 60, 0, 255).astype(np.uint8)
                    thermal[wy1:wy2, wx1:wx2] = bright
                thermal = draw_weapon_shape(thermal, wx1, wy1, wx2, wy2, t)
                pulse = int(abs(math.sin(t * 4)) * 100) + 155
                cv2.rectangle(thermal, (wx1 - 3, wy1 - 3), (wx2 + 3, wy2 + 3), (0, 0, pulse), 2)
                lx = min(wx2 + 8, w - 225)
                ly = wy1
                cv2.rectangle(thermal, (lx, ly), (lx + 222, ly + 58), (0, 0, 0), -1)
                cv2.rectangle(thermal, (lx, ly), (lx + 222, ly + 58), (0, 0, 140), 1)
                cv2.putText(thermal, "DETECTION: FIREARM_T01", (lx + 4, ly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 60, 255), 1)
                cv2.putText(thermal, "CONFIDENCE: 99.8%", (lx + 4, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 60, 255), 1)
                cv2.putText(thermal, "STATUS: ALERT", (lx + 4, ly + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 220), 1)
                pad = 18
                crop_y1 = max(0, wy1 - pad)
                crop_y2 = min(h, wy2 + pad)
                crop_x1 = max(0, wx1 - pad)
                crop_x2 = min(w, wx2 + pad)
                crop = thermal[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (200, 150))
                    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        weapon_crop_data = enc.tobytes()
    with weapon_jpg_lock:
        weapon_jpg = weapon_crop_data
    cv2.rectangle(thermal, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(thermal, "[CAM 01] CONTRABAND DETECTED | ZONE: ENTRY-HALL", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 220), 1)
    cv2.putText(thermal, "[SYSTEM] OPTIC/THERMAL | MODE: LIVE-MONITOR | NETWORK: OK", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1)
    return thermal

def draw_normal_hud(frame, fps, tracker, bag_box, bag_label, status):
    h, w = frame.shape[:2]
    if status == "CLEAR":
        color = (0, 200, 60)
    elif status == "BAG_DETECTED":
        color = (0, 140, 255)
    else:
        color = (0, 0, 210)
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(frame, "MURAQIB: " + status, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    cv2.putText(frame, "[SYSTEM] MODE: LIVE-MONITOR | NETWORK: OK", (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1)
    cv2.putText(frame, "FPS: " + str(int(fps)), (w - 80, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
    for pid in tracker:
        p = tracker[pid]
        if not p["in_frame"]:
            continue
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(frame, "Person " + str(pid), (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 100), 1)
    if bag_box and bag_label:
        bx1, by1, bx2, by2 = bag_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
        cv2.putText(frame, "BAG 1 " + bag_label, (bx1, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 1)
    ts = time.strftime("%Y.%m.%d %H:%M:%S")
    cv2.putText(frame, ts, (w - 200, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 80, 80), 1)
    return frame

def engine_loop():
    global latest_jpg, raw_frame
    print("Loading YOLO model")
    model = YOLO("yolo11s.pt")
    print("YOLO ready")
    face_det = load_face_detector()
    cam = Camera(0)
    tracker = Tracker()
    frame_n = 0
    fps_buf = deque(maxlen=24)
    t_prev = time.time()
    last_boxes = []
    bag_detected_at = None
    thermal_start_at = None
    last_bag_box = None
    bag_label = ""
    cv_tracker = None
    cv_tracker_ok = False
    color_t = 0
    log_event("System online")
    while True:
        frame = cam.read()
        with raw_frame_lock:
            raw_frame = frame.copy()
        frame_n += 1
        t_now = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        fps = float(np.mean(fps_buf))
        run_yolo = (frame_n % YOLO_EVERY_N == 0)
        if run_yolo:
            new_w = INPUT_WIDTH
            new_h = int(frame.shape[0] * new_w / frame.shape[1])
            small = cv2.resize(frame, (new_w, new_h))
            results = model(small, verbose=False, conf=CONF_THRESH)[0]
            scale_x = frame.shape[1] / small.shape[1]
            scale_y = frame.shape[0] / small.shape[0]
            last_boxes = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                last_boxes.append((int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y), cls, conf))
        person_boxes = []
        bag_detections = []
        for x1, y1, x2, y2, cls, conf in last_boxes:
            if cls == PERSON_CLASS:
                person_boxes.append((x1, y1, x2, y2))
            elif cls in BAG_CLASSES:
                bag_detections.append((x1, y1, x2, y2, cls))
        if run_yolo and bag_detections:
            bx1, by1, bx2, by2, bcls = bag_detections[0]
            bag_label = BAG_CLASSES[bcls]
            roi = (bx1, by1, bx2 - bx1, by2 - by1)
            try:
                cv_tracker = cv2.TrackerCSRT_create()
                cv_tracker.init(frame, roi)
                cv_tracker_ok = True
            except Exception:
                cv_tracker_ok = False
            last_bag_box = (bx1, by1, bx2, by2)
        elif cv_tracker_ok and cv_tracker is not None:
            ok, roi = cv_tracker.update(frame)
            if ok:
                rx, ry, rw, rh = [int(v) for v in roi]
                last_bag_box = (rx, ry, rx + rw, ry + rh)
            else:
                cv_tracker_ok = False
        if run_yolo and not bag_detections and not cv_tracker_ok:
            if bag_detected_at and t_now - bag_detected_at > 3:
                last_bag_box = None
                bag_label = ""
        if last_bag_box:
            if bag_detected_at is None:
                bag_detected_at = t_now
                log_event("Bag tracked: " + bag_label, "warn")
            if t_now - bag_detected_at >= BAG_DELAY and thermal_start_at is None:
                thermal_start_at = t_now
                log_event("Thermal scan: weapon detected", "threat")
        else:
            if bag_detected_at is not None:
                bag_detected_at = None
                log_event("Bag tracking lost")
            if thermal_start_at and t_now - thermal_start_at > THERMAL_DURATION:
                thermal_start_at = None
                log_event("Scan ended")
        thermal_active = thermal_start_at is not None
        if thermal_active:
            status = "THREAT"
        elif last_bag_box:
            status = "BAG_DETECTED"
        else:
            status = "CLEAR"
        people = tracker.update(person_boxes, frame.shape[1])
        if t_now - color_t > 1.2:
            color_t = t_now
            for pid in people:
                p = people[pid]
                if p["in_frame"]:
                    top, bot = get_clothing_colors(frame, p["box"])
                    p["color_top"] = top
                    p["color_bot"] = bot
        frame, face_active = blur_faces(frame, face_det, person_boxes)
        if thermal_active:
            frame = apply_thermal(frame, last_bag_box)
        else:
            frame = draw_normal_hud(frame, fps, people, last_bag_box, bag_label, status)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
        if ok:
            with latest_jpg_lock:
                latest_jpg = jpg.tobytes()
        persons_out = []
        for pid in people:
            p = people[pid]
            persons_out.append({"id": pid, "in_frame": p["in_frame"], "zone": p["zone"], "path": p["path"], "color_top": p["color_top"], "color_bot": p["color_bot"], "first": p["first"], "cx": p["cx"], "cy": p["cy"]})
        bag_info = None
        if last_bag_box:
            bag_cx = (last_bag_box[0] + last_bag_box[2]) // 2
            bag_info = {"label": bag_label, "zone": tracker.get_zone(bag_cx, frame.shape[1]), "box": list(last_bag_box)}
        with state_lock:
            state["status"] = status
            state["fps"] = round(fps, 1)
            state["persons"] = persons_out
            state["face_active"] = face_active
            state["thermal_active"] = thermal_active
            state["weapon_conf"] = 99.8 if thermal_active else 0
            state["bag"] = bag_info
            state["frame_count"] += 1

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.cors()
            self.end_headers()
            try:
                while True:
                    with latest_jpg_lock:
                        jpg = latest_jpg
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.030)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/state":
            with state_lock:
                data = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.cors()
            self.end_headers()
            self.wfile.write(data)
        elif path == "/weapon_crop":
            with weapon_jpg_lock:
                jpg = weapon_jpg
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.cors()
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_response(204)
                self.cors()
                self.end_headers()
        elif path.startswith("/person_crop/"):
            pid = int(path.split("/")[-1])
            with raw_frame_lock:
                rf = raw_frame.copy() if raw_frame is not None else None
            with state_lock:
                persons = state["persons"]
            person = None
            for p in persons:
                if p["id"] == pid:
                    person = p
                    break
            if rf is not None and person:
                cx = person["cx"]
                cy = person["cy"]
                s = 80
                x1 = max(0, cx - s)
                y1 = max(0, cy - s * 2)
                x2 = min(rf.shape[1], cx + s)
                y2 = min(rf.shape[0], cy + int(s * 1.5))
                crop = rf[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (120, 180))
                    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.cors()
                        self.end_headers()
                        self.wfile.write(enc.tobytes())
                        return
            self.send_response(204)
            self.cors()
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        if self.path == "/reset":
            with state_lock:
                state["status"] = "CLEAR"
                state["thermal_active"] = False
                state["weapon_conf"] = 0
            log_event("Alert reset by operator")
            self.send_response(200)
            self.cors()
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def main():
    print("Starting Muraqib")
    t = threading.Thread(target=engine_loop, daemon=True)
    t.start()
    time.sleep(3)
    print("Stream:  http://YOUR-PI-IP:" + str(STREAM_PORT) + "/stream")
    print("Open dashboard.html in your browser")
    server = HTTPServer(("0.0.0.0", STREAM_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main()
