import subprocess
import os

def test_adb_connection():
    print("Testing Android Debug Bridge (ADB) Connection...")
    
    # HARDCODED PATH TO ADB FOR YOUR SPECIFIC WINDOWS PC
    adb_path = r"C:\Users\s\AppData\Local\Android\Sdk\platform-tools\adb.exe"
    
    if not os.path.exists(adb_path):
        print(f"❌ Could not find ADB at {adb_path}")
        print("Please check if Android Studio is installed!")
        return

    try:
        # Use the hardcoded path instead of just "adb"
        result = subprocess.run([adb_path, "devices"], capture_output=True, text=True, check=True)
        
        output_lines = result.stdout.strip().split('\n')
        
        if len(output_lines) > 1:
            print("\n✅ SUCCESS! Python successfully detected your Android device/emulator.")
            print("Device List:")
            print(result.stdout)
            print("We are cleared to begin Phase 3 integration!")
        else:
            print("\n❌ ADB is working, but no device was detected.")
            print("Make sure your phone is plugged in and USB Debugging is ON.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_adb_connection()