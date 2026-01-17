import time
import os
import sys

DASHBOARD_PATH = "training_dashboard.md"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def view_progress():
    print("Waiting for dashboard updates... (Press Ctrl+C to stop)")
    try:
        while True:
            if os.path.exists(DASHBOARD_PATH):
                with open(DASHBOARD_PATH, "r") as f:
                    content = f.read()
                
                clear_screen()
                print("==========================================")
                print("   SENTINEL TRAINING PROGRESS VIEWER")
                print("==========================================\n")
                print(content)
                print("\n==========================================")
                print("Last checked: " + time.strftime("%H:%M:%S"))
            else:
                clear_screen()
                print(f"Dashboard file '{DASHBOARD_PATH}' not found yet. waiting...")
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting viewer.")
        sys.exit(0)

if __name__ == "__main__":
    view_progress()
