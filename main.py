from detector.hand_detector import HandDetector
from detector.gesture_controller import GestureController


def main():
    hand_detector = HandDetector()
    if not hand_detector.detector:
        print("Warning: No trained detector loaded. You may need to train first.")

    gesture_ctrl = GestureController(hand_detector.detector)

    while True:
        print("\n=== Hand Gesture Control Menu ===")
        print("1. Collect Training Data")
        print("2. Train Detector")
        print("3. Calibrate Gestures")
        print("4. Test Detector")
        print("5. Run Game Controller")
        print("0. Exit")

        choice = input("Choose option: ")

        if choice == '1':
            try:
                num = int(input("Number of samples to collect 100- 200: "))
                clean = input("Clean existing dataset? (y/n): ").lower() == 'y'
                hand_detector.collect_data(num_samples=num, clean_dataset=clean)
            except ValueError:
                print("Invalid number.")
        elif choice == '2':
            hand_detector.train_detector()
        elif choice == '3':
            gesture_ctrl.calibrate()
        elif choice == '4':
            hand_detector.test_detector()
        elif choice == '5':
            gesture_ctrl.detect_and_control()
        elif choice == '0':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
