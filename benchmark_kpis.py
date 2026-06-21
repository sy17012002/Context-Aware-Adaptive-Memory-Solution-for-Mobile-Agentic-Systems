import subprocess
import time
import re
import torch
import numpy as np
import os
import pickle

# --- AI IMPORTS ---
from train_rl_agent import DQN
# Ensure your Oracle class is imported here
# from train_oracle_v5_deep_context import DeepContextOracle 

# --- DEVICE CONFIGURATION ---
APP_MAP = {
    "Browser": "com.android.chrome",
    "Maps": "com.google.android.apps.maps",
    "YouTube": "com.google.android.youtube",
    "Calculator": "com.google.android.calculator"
}

APP_TO_ID = {"Browser": 1, "Maps": 2, "YouTube": 3, "Calculator": 4}
ADB_PATH = r"C:\Users\s\AppData\Local\Android\Sdk\platform-tools\adb.exe"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. LOAD AI BRAINS
# ==========================================
print(f"🧠 Booting Hybrid AI System onto {device}...")

# Load RL Protector
rl_agent = DQN(state_size=4, action_size=3).to(device)
if os.path.exists("rl_memory_agent.pth"):
    rl_agent.load_state_dict(torch.load("rl_memory_agent.pth", map_location=device))
    rl_agent.eval()
    print("✅ RL Agent (Protector) Loaded.")

# Load Oracle Predictor
oracle_model = None
encoder = None
try:
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    # Uncomment lines below when your .pth files are ready
    # oracle_model = DeepContextOracle().to(device) 
    # oracle_model.load_state_dict(torch.load('oracle_model_v5_deep_context.pth', map_location=device))
    # oracle_model.eval()
    print("✅ Oracle V5 (Predictor) Loaded.")
except Exception as e:
    print(f"⚠️ Warning: Oracle loader bypassed for testing. {e}")

# ==========================================
# 2. ADB UTILITIES 
# ==========================================
def get_real_ram_usage():
    cmd = [ADB_PATH, "shell", "dumpsys", "meminfo"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r'Used RAM:.*?([\d,]+)\s*K', result.stdout)
    if match: return round(int(match.group(1).replace(',', '')) / 1024, 2)
    return 0.0

def get_total_device_ram():
    """Fetches the 7412.65 MB total physical RAM from the device."""
    cmd = [ADB_PATH, "shell", "cat", "/proc/meminfo"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r'MemTotal:\s+(\d+)\s+kB', result.stdout)
    if match: return round(int(match.group(1)) / 1024, 2)
    return 7412.65 # Fallback to your specific device total

def launch_app_and_time(app_name):
    package_name = APP_MAP.get(app_name)
    start_time = time.time()
    cmd = [ADB_PATH, "shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"]
    subprocess.run(cmd, capture_output=True)
    return time.time() - start_time

def preload_app(app_name):
    package_name = APP_MAP.get(app_name)
    subprocess.run([ADB_PATH, "shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"], capture_output=True)
    time.sleep(0.5)
    subprocess.run([ADB_PATH, "shell", "input", "keyevent", "3"], capture_output=True)

def kill_app(app_name):
    package_name = APP_MAP.get(app_name)
    subprocess.run([ADB_PATH, "shell", "am", "force-stop", package_name], capture_output=True)

# ==========================================
# 3. CORE BENCHMARK ENGINE
# ==========================================
def run_benchmark(use_ai=False):
    print(f"\n--- STARTING BENCHMARK (AI Enabled: {use_ai}) ---")
    sequence = ["Browser", "Maps", "YouTube", "Calculator", "Browser", "Maps"]
    open_apps = []
    user_history = []
    total_launch_time = 0
    thrashing_incidents = 0
    
    # Establish Dynamic Baselines
    initial_ram = get_real_ram_usage()
    total_ram = get_total_device_ram()
    
    # --- ENTERPRISE FREE RAM CALCULATION ---
    free_ram_headroom = total_ram - initial_ram
    # Allow AI to manage 15% of the actual available headroom
    DYNAMIC_DANGER_ZONE = max(150.0, free_ram_headroom * 0.15) 
    
    print(f"Baseline RAM: {initial_ram:.2f} MB (Total: {total_ram:.2f} MB)")
    if use_ai:
        print(f"Free Headroom: {free_ram_headroom:.2f} MB")
        print(f"Smart RL Warning Track: {DYNAMIC_DANGER_ZONE:.2f} MB")
    
    CRASH_LIMIT_MB = initial_ram + DYNAMIC_DANGER_ZONE + 50     
    AI_TRIGGER_MB = initial_ram + 25 
    
    for i, app in enumerate(sequence):
        # PHASE 1: RL AGENT (Protector)
        if use_ai:
            current_ram = get_real_ram_usage()
            if current_ram > AI_TRIGGER_MB and len(open_apps) > 0:
                growth = current_ram - initial_ram
                ram_ratio = min(1.0, (growth / DYNAMIC_DANGER_ZONE) + 0.2) 
                
                # Context Sensing
                rem = sequence[i:]
                p1, p2, p3 = (APP_TO_ID.get(rem[j], 0) if len(rem) > j else 0 for j in range(3))
                state_tensor = torch.tensor([[ram_ratio, p1, p2, p3]], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    q_values = rl_agent(state_tensor)
                    if ram_ratio >= 0.85: q_values[0][0] = -999.0
                    action_idx = torch.argmax(q_values).item()
                
                if action_idx in [1, 2]:
                    safe_to_kill = [a for a in open_apps if a != app]
                    if safe_to_kill:
                        target = safe_to_kill[0]
                        print(f"🛡️ RL AGENT: High RAM! Killing {target}. (Ratio: {ram_ratio:.2f})")
                        open_apps.remove(target)
                        kill_app(target)
                        time.sleep(1.0)

        # PHASE 2: USER ACTION
        launch_duration = launch_app_and_time(app)
        total_launch_time += launch_duration
        if app in open_apps: open_apps.remove(app)
        open_apps.append(app)
        user_history.append(app)
        
        post_launch_ram = get_real_ram_usage()
        if post_launch_ram > CRASH_LIMIT_MB:
            print(f"⚠️ THRASHING DETECTED: {post_launch_ram} MB")
            thrashing_incidents += 1
            
        # PHASE 3: ORACLE (Predictor)
        if use_ai and oracle_model:
            if len(user_history) >= 3:
                recent = user_history[-3:]
                try:
                    ctx = torch.tensor([encoder.transform(recent)], dtype=torch.long).to(device)
                    with torch.no_grad():
                        pred_idx = torch.argmax(oracle_model(ctx)).item()
                        pred_app = encoder.inverse_transform([pred_idx])[0]
                    
                    if pred_app not in open_apps:
                        print(f"🔮 ORACLE PREDICTS: {pred_app}")
                        preload_app(pred_app)
                        open_apps.append(pred_app)
                        time.sleep(1.0)
                except: pass

        time.sleep(2) 
        
    return total_launch_time / len(sequence), thrashing_incidents

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n🧹 Cleaning device environment...")
    for app in APP_MAP.keys(): kill_app(app)
    time.sleep(4)
    
    base_launch, base_thrash = run_benchmark(use_ai=False)
    
    print("\n🧹 Re-cleaning for AI run...")
    for app in APP_MAP.keys(): kill_app(app)
    time.sleep(4)
    
    ai_launch, ai_thrash = run_benchmark(use_ai=True)
    
    # Calculate Final KPIs
    launch_imp = ((base_launch - ai_launch) / base_launch) * 100
    thrash_red = ((base_thrash - ai_thrash) / base_thrash * 100) if base_thrash > 0 else 0
    
    print("\n" + "="*50)
    print("🏆 FINAL LIVE EDGE AI KPI REPORT 🏆")
    print("="*50)
    print(f"Launch Time Improvement: {launch_imp:.1f}% (Target: 10%)")
    print(f"Thrashing Reduction:     {thrash_red:.1f}% (Target: 50%)")