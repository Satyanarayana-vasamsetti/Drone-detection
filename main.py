import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load a pretrained YOLO model
model = YOLO("C:/Users/satya/PycharmProjects/Object_Detection/best (1).pt")

# Open a connection to the webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the application window name
app_name = "Drone Detection"

# Create a figure for Matplotlib
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# Set the figure title
fig.canvas.manager.set_window_title(app_name)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform object detection
        results = model(frame)

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame using Matplotlib
        ax.imshow(frame_rgb)
        ax.axis('off')  # Hide axes for a cleaner look

        # Annotate each bounding box with coordinates and class name
        for result in results:
            boxes = result.boxes  # Extract bounding boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Extract box coordinates and convert to numpy array
                x1, y1, x2, y2 = map(int, xyxy)  # Convert to integers
                confidence = box.conf.item()  # Confidence score
                class_id = int(box.cls.item())  # Class ID
                class_name = model.names[class_id]  # Get class name from the model
                label = f"{class_name} ({x1}, {y1}), ({x2}, {y2})"

                # Draw the bounding box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='green', facecolor='none')
                ax.add_patch(rect)

                # Display the class name and coordinates near the bounding box
                ax.text(x1 + 10, y1 - 10, label, color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

        plt.pause(0.001)  # Pause to update the plot
        ax.clear()  # Clear the previous frame

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # When everything is done, release the capture and close windows
    cap.release()
    plt.close(fig)
    cv2.destroyAllWindows()
