import time
import argparse
import os

class trigger:
    def Prepare(self):
        Cam1Triggerred = False
        Cam1TriggerredTime = 0
        UltraSonicTriggerred = False
        UltraSonicTriggerredTime = 0

    def triggerCam1(self):
        self.Cam1Triggerred = True
        self.Cam1TriggerredTime = time.time()
    
    def triggerUltraSonic(self):
        self.UltraSonicTriggerred = True
        self.UltraSonicTriggerredTime = time.time()

    def GetTriggerStatus(self):
        return {
            "Cam1": {
                "Triggered": self.Cam1Triggerred,
                "Time": self.Cam1TriggerredTime
            },
            "UltraSonic": {
                "Triggered": self.UltraSonicTriggerred,
                "Time": self.UltraSonicTriggerredTime
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trigger class")
    parser.add_argument("--entity", 
                        type=str, 
                        default="Cam1",
                        help="YOU GUESS?")
    args = parser.parse_args()

    trigger_instance = trigger()
    if args.entity == "Cam1":
        trigger_instance.triggerCam1()
        print(f"Cam1 Triggered: {trigger_instance.Cam1Triggerred}, Time: {trigger_instance.Cam1TriggerredTime}")
    elif args.entity == "UltraSonic":
        trigger_instance.triggerUltraSonic()
        print(f"UltraSonic Triggered: {trigger_instance.UltraSonicTriggerred}, Time: {trigger_instance.UltraSonicTriggerredTime}")
    else:
        print("Error: Unknown entity.")
