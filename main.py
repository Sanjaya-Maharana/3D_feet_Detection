from flask import Flask, render_template, request
import os
import cv2
import mediapipe as mp
app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
def empty_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))


@app.route('/')
def single_upload():
    return render_template('home.html')

@app.route('/', methods=['GET','POST'])
def single():
    data = {}
    empty_directory('Shoe')
    empty_directory('static')
    file = request.files['file']
    file.save('Shoe/' + file.filename)
    image = cv2.imread('Shoe/' + file.filename)

    # Resize the image to half its original size
    resized_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Initialize Objectron model
    with mp_objectron.Objectron(static_image_mode=True, max_num_objects=5,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5,
                                model_name='Shoe') as objectron:

        # Process the resized image with Objectron
        results = objectron.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

        # Annotate the image with the results
        annotated_image = resized_image.copy()
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                                     detected_object.translation)

        # Save the detected image to disk
        annotated_image_path = 'static/annotated_image.jpg'
        cv2.imwrite(annotated_image_path, annotated_image)
    return render_template('results.html', image_path='annotated_image.jpg')




if __name__ == '__main__':
    app.run(debug=True)