from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import time
import threading
import requests
import io

app = Flask(__name__)

# --- CONFIGURATION ---
TELEGRAM_TOKEN = "8894336150:AAFebBaORi8C8471hNep_Lg9iTsxVDddV9o"
TELEGRAM_CHAT_ID = "1899583396"

# Global states
camera = None
chat_history = []
lock = threading.Lock()
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
        camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    return camera

# --- CENTRALIZED SYSTEM LOGGER ---
def post_system_log(message_text, emoji="⚙️"):
    formatted_web_msg = f"<b>{emoji} [SYSTEM]:</b> {message_text}"
    formatted_tg_msg = f"{emoji} *[SYSTEM]*\n{message_text}"
    
    with lock:
        chat_history.append({"sender": "bot", "text": formatted_web_msg})
    send_telegram_msg(formatted_tg_msg)

# Helper to push text messages to Telegram
def send_telegram_msg(text):
    clean_text = text.replace("<b>", "*").replace("</b>", "*").replace("<code>", "`").replace("</code>", "`")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": clean_text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
         print(f"Failed to push message to Telegram: {e}")

# Helper to push photos to Telegram
def send_telegram_photo(image_bytes, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('capture.jpg', image_bytes, 'image/jpeg')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
    try:
        requests.post(url, files=files, data=data, timeout=10)
    except Exception as e:
        print(f"Failed to push photo to Telegram: {e}")

# --- BACKGROUND TELEGRAM POLLER ---
def telegram_listener():
    offset = 0
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    
    post_system_log("System successfully initialized. Watching Telegram for commands...", "🟢")
    
    while True:
        try:
            url = f"{base_url}/getUpdates?offset={offset}&timeout=10"
            r = requests.get(url, timeout=15)
            data = r.json()
            
            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    offset = update["update_id"] + 1
                    
                    msg_obj = update.get("message")
                    if msg_obj and str(msg_obj["chat"]["id"]) == str(TELEGRAM_CHAT_ID):
                        text = msg_obj.get("text", "")
                        
                        if text:
                            if text.lower() in ["analyze", "/analyze"]:
                                post_system_log("Running Topdon thermal analysis scan (via Telegram)...", "🔄")
                                reply = run_thermal_analysis()
                                post_system_log(reply, "📊")
                            elif text.lower() in ["capture", "/capture"]:
                                post_system_log("Capturing live visual frame for Telegram upload...", "📸")
                                run_image_capture()
                            elif text.lower() in ["temp", "/temp"]:
                                reply = get_cpu_temp()
                                post_system_log(reply, "🌡️")
                            else:
                                with lock:
                                    chat_history.append({"sender": "user", "text": f"<b>[Telegram Phone]:</b> {text}"})
                                    
        except Exception as e:
            print(f"[Telegram Error]: {e}")
            
        time.sleep(1)

# --- VISUAL CAPTURE COMMAND ENGINE ---
def run_image_capture():
    try:
        with camera_lock:
            cap = get_camera()
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1) # Turn on RGB mode to get clear visual colors
            time.sleep(0.05)
            ret, frame = cap.read()
            
        if not ret:
            post_system_log("Capture failed: Could not read camera buffer.", "❌")
            return
            
        # Crop to the top visual portion of the frame
        if len(frame.shape) == 3:
            visual_crop = frame[0:192, 0:256]
        else:
            visual_crop = frame
            
        # Standardize colormap layout and size
        color_frame = cv2.applyColorMap(visual_crop, cv2.COLORMAP_JET)
        resized_frame = cv2.resize(color_frame, (480, 360))
        
        # Encode into in-memory JPEG byte array
        success, buffer = cv2.imencode('.jpg', resized_frame)
        if success:
            send_telegram_photo(buffer.tobytes(), caption="📸 Topdon Live Snapshot")
        else:
            post_system_log("Capture failed: JPEG encoding error.", "❌")
    except Exception as e:
        post_system_log(f"Capture execution error: {str(e)}", "❌")

# --- THERMAL PROCESSING AND ANALYSIS ---
def run_thermal_analysis():
    try:
        with camera_lock:
            cap = get_camera()
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) # Turn off RGB mode for pure raw bits
            time.sleep(0.05)
            ret, frame = cap.read()
            if not ret:
                return "Error: Could not pull raw frame buffer from camera."
            
            raw_data = np.frombuffer(frame, dtype=np.uint8)
        
        if raw_data.size == 147456:
            reshaped = raw_data.reshape((384, 384))
            thermal_slice = reshaped[192:384, 0:256].astype(np.float32)
        elif raw_data.size == 196608 or raw_data.size == 98304:
            raw_data_16bit = np.frombuffer(frame, dtype=np.uint16)
            reshaped = raw_data_16bit.reshape((384, 256))
            thermal_slice = reshaped[192:384, :].astype(np.float32)
        elif raw_data.size == 294912:
            reshaped_rgb = raw_data.reshape((384, 256, 3))
            thermal_slice = reshaped_rgb[192:384, :, 0].astype(np.float32)
        else:
            return f"Error: Unexpected byte array size ({raw_data.size})."

        celsius_matrix = (thermal_slice / 64.0) - 273.15
        if np.max(celsius_matrix) < -50 or np.max(celsius_matrix) > 200:
            celsius_matrix = (thermal_slice / 100.0) - 273.15

        max_temp = np.max(celsius_matrix)
        min_temp = np.min(celsius_matrix)
        avg_temp = np.mean(celsius_matrix)
        
        status = "HOT (Warning)" if max_temp > 45.0 else "Normal/Temperate"
        
        return (f"Topdon Thermal Results:\n"
                f"• Target Status: {status}\n"
                f"• Max Temp: {max_temp:.1f}°C\n"
                f"• Min Temp: {min_temp:.1f}°C\n"
                f"• Avg Temp: {avg_temp:.1f}°C")
                
    except Exception as e:
        return f"Thermal scan run failed: {str(e)}"

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000
        return f"Orange Pi CPU Temperature: {temp:.1f}°C"
    except:
        return "Failed to access internal CPU temperature hardware."

def gen_frames():
    while True:
        with camera_lock:
            cap = get_camera()
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1) 
            ret, frame = cap.read()
            
        if not ret:
            time.sleep(0.1)
            continue
            
        if len(frame.shape) == 3:
            visual_crop = frame[0:192, 0:256]
        else:
            visual_crop = frame
            
        color_frame = cv2.applyColorMap(visual_crop, cv2.COLORMAP_JET)
        resized_frame = cv2.resize(color_frame, (480, 360))
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

# --- ROUTING SYSTEM ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_messages')
def get_messages():
    with lock:
        return jsonify({"history": chat_history})

@app.route('/send_message', methods=['POST'])
def send_message():
    user_data = request.get_json()
    msg = user_data.get("message", "").strip()
    
    if msg:
        if msg.lower() in ["analyze", "/analyze"]:
            post_system_log("Running Topdon thermal analysis scan (via PC Dashboard)...", "🔄")
            reply = run_thermal_analysis()
            post_system_log(reply, "📊")
            
        elif msg.lower() in ["capture", "/capture"]:
            post_system_log("Capturing live visual frame (via PC Dashboard)...", "📸")
            run_image_capture()
            
        elif msg.lower() in ["temp", "/temp"]:
            reply = get_cpu_temp()
            post_system_log(reply, "🌡️")
            
        else:
            with lock:
                chat_history.append({"sender": "user", "text": f"<b>[Web Dashboard]:</b> {msg}"})
            send_telegram_msg(msg)
        
    return jsonify({"status": "processed"})

if __name__ == '__main__':
    t = threading.Thread(target=telegram_listener, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)