import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Define video capture object
cap = cv2.VideoCapture('WhatsApp Video 2023-04-06 at 09.55.40.mp4')

# Define video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Initialize Objectron model
with mp_objectron.Objectron(static_image_mode=False, max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_name='Shoe') as objectron:

    # Set start time
    start_time = time.time()

    # Loop through each frame of the video
    while cap.isOpened():

        # Check if elapsed time has exceeded 50 seconds
        if (time.time() - start_time) > 50:
            break

        # Read the frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break out of the loop
        if not ret:
            break

        # Resize the frame to reduce its size
        frame = cv2.resize(frame, (640, 480))

        # Process the frame with Objectron
        results = objectron.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Copy the frame and annotate with the results
        annotated_frame = frame.copy()
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    annotated_frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(annotated_frame, detected_object.rotation,
                                     detected_object.translation)

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame in a window
        cv2.imshow('Objectron Shoe Detection', annotated_frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and writer objects
    cap.release()
    out.release()

    # Close the OpenCV windows
    cv2.destroyAllWindows()
