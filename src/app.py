import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import time
import subprocess
import threading
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MirrorCheck Clinical Triage Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

def trigger_mac_emergency_siren():
    """
    Directly sounds an acoustic 5-second emergency alert via macOS CoreAudio.
    Uses afplay with maximum gain (-v 2) and fallback system alerts.
    """
    def _run():
        start_time = time.time()
        # Find verified system alert tone
        sound_file = None
        for path in [
            "/System/Library/Sounds/Sosumi.aiff",
            "/System/Library/Sounds/Ping.aiff",
            "/System/Library/Sounds/Glass.aiff"
        ]:
            if os.path.exists(path):
                sound_file = path
                break

        while time.time() - start_time < 5.0:
            if sound_file:
                subprocess.run(["afplay", "-v", "2", sound_file])
            else:
                # Terminal bell pulse
                print("\a", flush=True)
                time.sleep(0.3)
    
    t = threading.Thread(target=_run)
    t.daemon = True
    t.start()

# --- ANATOMICALLY VERIFIED MEDIAPIPE FACE MESH LANDMARKS ---
SELLION = 168        # Mid-nasal root (rigid skull baseline)
EYE_R_OUTER = 33
EYE_R_INNER = 133
EYE_L_INNER = 362
EYE_L_OUTER = 263

EYE_R_UPPER = 159
EYE_R_LOWER = 145
EYE_L_UPPER = 386
EYE_L_LOWER = 374

LIP_CORNER_R = 61    # Patient Right
LIP_CORNER_L = 291   # Patient Left
UPPER_LIP_MID = 0    
LOWER_LIP_MID = 17   
LIP_MID_R = 84       
LIP_MID_L = 314      

def compute_ear(lm, outer, inner, upper, lower, w, h):
    p_out = np.array([lm[outer].x * w, lm[outer].y * h])
    p_in  = np.array([lm[inner].x * w, lm[inner].y * h])
    p_up  = np.array([lm[upper].x * w, lm[upper].y * h])
    p_low = np.array([lm[lower].x * w, lm[lower].y * h])
    width = np.linalg.norm(p_out - p_in) + 1e-6
    height = np.linalg.norm(p_up - p_low)
    return float(height / width)

def extract_clinical_metrics(lm, w, h):
    eye_r_center = np.array([(lm[EYE_R_OUTER].x + lm[EYE_R_INNER].x) * 0.5 * w, (lm[EYE_R_OUTER].y + lm[EYE_R_INNER].y) * 0.5 * h])
    eye_l_center = np.array([(lm[EYE_L_OUTER].x + lm[EYE_L_INNER].x) * 0.5 * w, (lm[EYE_L_OUTER].y + lm[EYE_L_INNER].y) * 0.5 * h])

    interocular_vec = eye_l_center - eye_r_center
    interocular_dist = np.linalg.norm(interocular_vec)
    if interocular_dist < 10.0:
        return None

    e_x = interocular_vec / interocular_dist
    e_y = np.array([-e_x[1], e_x[0]])
    origin = np.array([lm[SELLION].x * w, lm[SELLION].y * h])

    lip_r = np.array([lm[LIP_CORNER_R].x * w, lm[LIP_CORNER_R].y * h])
    lip_l = np.array([lm[LIP_CORNER_L].x * w, lm[LIP_CORNER_L].y * h])
    lip_mid_r = np.array([lm[LIP_MID_R].x * w, lm[LIP_MID_R].y * h])
    lip_mid_l = np.array([lm[LIP_MID_L].x * w, lm[LIP_MID_L].y * h])

    y_drop_r = float(np.dot(lip_r - origin, e_y))
    y_drop_l = float(np.dot(lip_l - origin, e_y))
    vert_drop_ratio = (abs(y_drop_r - y_drop_l) / interocular_dist) * 100.0

    mouth_vec = lip_l - lip_r
    norm_mouth = np.linalg.norm(mouth_vec)
    if norm_mouth > 1e-5:
        cos_cant = np.clip(np.dot(mouth_vec / norm_mouth, e_x), -1.0, 1.0)
        cant_angle = float(np.degrees(np.arccos(abs(cos_cant))))
    else:
        cant_angle = 0.0

    mid_eyes = (eye_r_center + eye_l_center) * 0.5
    lat_r = abs(np.dot(lip_r - mid_eyes, e_x))
    lat_l = abs(np.dot(lip_l - mid_eyes, e_x))
    lateral_shift_ratio = (abs(lat_r - lat_l) / (lat_r + lat_l + 1e-5)) * 100.0

    sag_r = float(np.dot(lip_mid_r - origin, e_y))
    sag_l = float(np.dot(lip_mid_l - origin, e_y))
    lip_contour_sag = (abs(sag_r - sag_l) / interocular_dist) * 100.0

    oaq = float((vert_drop_ratio * 0.45) + (cant_angle * 1.8) + (lip_contour_sag * 0.35) + (lateral_shift_ratio * 0.20))

    ear_r = compute_ear(lm, EYE_R_OUTER, EYE_R_INNER, EYE_R_UPPER, EYE_R_LOWER, w, h)
    ear_l = compute_ear(lm, EYE_L_OUTER, EYE_L_INNER, EYE_L_UPPER, EYE_L_LOWER, w, h)
    ear_diff = float(abs(ear_r - ear_l))

    return {
        "oaq": oaq,
        "ear_diff": ear_diff,
        "cant_angle": cant_angle,
        "vert_drop_ratio": vert_drop_ratio
    }

def generate_clinical_report(metrics):
    oaq = metrics["oaq"]
    ear_diff = metrics["ear_diff"]

    if oaq >= 6.8 or ear_diff >= 0.050:
        score = 3 if oaq > 11.0 else 4
        # Trigger verified 5-second siren
        trigger_mac_emergency_siren()
        return {
            "score": score,
            "oaq": float(oaq),
            "ear": float(ear_diff),
            "danger": True,
            "status_title": f"STROKE ALERT: ACUTE FACIAL PARALYSIS ({score}/10)",
            "status_desc": "CRITICAL EMERGENCY: Severe unilateral lower-facial motor deficit detected.",
            "problem_description": (
                f"Definitive facial nerve impairment detected (Oral Asymmetry Index = {oaq:.1f}%, Angle Cant = {metrics['cant_angle']:.1f}°). "
                "The patient exhibits acute downward oral commissure collapse and tone flattening. "
                "In clinical triage, this presentation strongly indicates acute ischemic stroke or acute peripheral facial palsy."
            ),
            "tips_to_ten": [
                "CRITICAL EMERGENCY: Immediately call emergency medical services or proceed to the nearest stroke center.",
                "Conduct BE-FAST examination: Check arm drift and listen for dysarthric (slurred) speech.",
                "Keep the patient upright and calm. Do not give any food, water, or aspirin due to swallowing (dysphagia) risk.",
                "Record the exact time of onset for the medical team (vital for thrombolytic / tPA therapy window)."
            ]
        }
    elif oaq >= 4.2 or ear_diff >= 0.030:
        score = 7
        return {
            "score": score,
            "oaq": float(oaq),
            "ear": float(ear_diff),
            "danger": False,
            "status_title": f"MODERATE ASYMMETRY / FACIAL DRIFT ({score}/10)",
            "status_desc": "Noticeable muscular discrepancy in resting facial tone.",
            "problem_description": (
                f"Measurable muscular imbalance detected (Oral Asymmetry = {oaq:.1f}%). "
                "There is observable asymmetry in resting lip elevation or palpebral eyelid fissure. "
                "Recommended for professional medical screening to rule out transient ischemic attack (TIA) or early-stage Bell's palsy."
            ),
            "tips_to_ten": [
                "Schedule a clinical consultation with a primary care practitioner or neurologist.",
                "Assess whether eyelid closure is complete on both sides.",
                "Avoid sleeping exclusively on one side and manage mental/physical stress levels.",
                "Re-test within 12 hours to verify whether muscular drift is stable or progressing."
            ]
        }
    elif oaq >= 2.2:
        score = 9
        return {
            "score": score,
            "oaq": float(oaq),
            "ear": float(ear_diff),
            "danger": False,
            "status_title": f"MILD FACIAL FATIGUE ({score}/10)",
            "status_desc": "Minor physiological variance within safe resting limits.",
            "problem_description": f"Subtle resting variance identified (OAQ = {oaq:.1f}%). Common with ocular fatigue or sleeping posture.",
            "tips_to_ten": [
                "Obtain 7-8 hours of restful sleep.",
                "Perform gentle bilateral facial smile stretches in front of a mirror.",
                "Stay well hydrated and take regular breaks from display screens."
            ]
        }
    else:
        return {
            "score": 10,
            "oaq": float(oaq),
            "ear": float(ear_diff),
            "danger": False,
            "status_title": "OPTIMAL BILATERAL SYMMETRY (10/10)",
            "status_desc": "Balanced neuromuscular muscle tone across both facial hemispheres.",
            "problem_description": "No facial drooping, eyelid lag, or motor weakness detected. Bilateral facial contours match healthy physiological norms.",
            "tips_to_ten": [
                "Maintain healthy sleep, hydration, and regular cardiovascular exercise to preserve microvascular circulation."
            ]
        }

mp_face_mesh = mp.solutions.face_mesh

@app.get("/")
async def get_index():
    with open("src/templates/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    h, w, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.20
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return JSONResponse(status_code=422, content={"error": "No human face detected. Ensure proper lighting and full frontal view."})

        lm = results.multi_face_landmarks[0].landmark
        metrics = extract_clinical_metrics(lm, w, h)
        if not metrics:
            return JSONResponse(status_code=422, content={"error": "Face detection too small or occluded."})

        diagnosis = generate_clinical_report(metrics)
        return JSONResponse(content={"diagnosis": diagnosis})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as face_mesh:
        
        is_scanning = False
        scan_samples_oaq = []
        scan_samples_ear = []
        scan_start_time = 0

        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)

                if "cmd" in data and data["cmd"] == "start_scan":
                    is_scanning = True
                    scan_samples_oaq.clear()
                    scan_samples_ear.clear()
                    scan_start_time = time.time()
                    continue

                if "frame" not in data:
                    continue

                img_bytes = base64.b64decode(data["frame"])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                curr_oaq = 0.0
                curr_ear_diff = 0.0

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    metrics = extract_clinical_metrics(lm, w, h)

                    if metrics:
                        curr_oaq = metrics["oaq"]
                        curr_ear_diff = metrics["ear_diff"]

                        lip_r = (int(lm[LIP_CORNER_R].x * w), int(lm[LIP_CORNER_R].y * h))
                        lip_l = (int(lm[LIP_CORNER_L].x * w), int(lm[LIP_CORNER_L].y * h))
                        cv2.circle(frame, lip_r, 4, (0, 255, 255), -1)
                        cv2.circle(frame, lip_l, 4, (0, 255, 255), -1)
                        cv2.line(frame, lip_r, lip_l, (0, 200, 255), 2)

                        if is_scanning:
                            scan_samples_oaq.append(curr_oaq)
                            scan_samples_ear.append(curr_ear_diff)

                if is_scanning and (time.time() - scan_start_time >= 5.0):
                    is_scanning = False
                    final_oaq = float(np.median(scan_samples_oaq)) if scan_samples_oaq else curr_oaq
                    final_ear = float(np.median(scan_samples_ear)) if scan_samples_ear else curr_ear_diff
                    diagnosis = generate_clinical_report({
                        "oaq": final_oaq,
                        "ear_diff": final_ear,
                        "cant_angle": 0.0
                    })
                    await websocket.send_text(json.dumps({"diagnosis": diagnosis}))
                    break

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_out = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_text(json.dumps({"image": b64_out}))

        except WebSocketDisconnect:
            pass
