# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%                                               %%%%%
# %%%%%          BISMILLAH HIRRAHMA NIRRAHEEM         %%%%%
# %%%%%                                               %%%%%
# %%%%%         Programmed By: Muzammil Behzad        %%%%%
# %%%%% Center for Machine Vision and Signal Analysis %%%%%
# %%%%%              University of Oulu               %%%%%
# %%%%%                 Oulu, Finland                 %%%%%
# %%%%%                                               %%%%%
# %%%%%        Email: muzammil.behzad@oulu.fi         %%%%%
# %%%%%                                               %%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import cv2
import datetime
import mediapipe as mp

# import useful function
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# create the haar cascade for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize camera capture
video = cv2.VideoCapture(0)

flip_image = True
resize_image = True
while True:
    ret, image = video.read()
    if resize_image:
        x_print, y_print, _ = image.shape
        extend_size = 2.5
        image = cv2.resize(image, (int(extend_size*y_print), int(extend_size*x_print)))
        x_print, y_print, _ = image.shape
        
        x_print = int(0.15*x_print)
        y_print = int(0.05*y_print)
        text_size = 2
        text_width = 6
        color_to_print = (0,255,255)
    else:
        x_print, y_print, _ = image.shape
        x_print = int(0.23*x_print)
        y_print = int(0.05*y_print)
        text_size = 0.7
        text_width = 2
        color_to_print = (0,0,255)

    if flip_image:
        image= cv2.flip(image, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} face(s)!".format(len(faces)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if len(faces) != 0:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=len(faces),
                                min_tracking_confidence=0.5) as face_mesh :

            image.flags.writeable = False
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=image,
                                            landmark_list=face_landmarks,
                                            connections=mp_face_mesh.FACE_CONNECTIONS,
                                            landmark_drawing_spec=drawing_spec,
                                            connection_drawing_spec=drawing_spec)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    now_time = datetime.datetime.now()
    window_title = 'Face Mesh Detection by Muzammil Behzad'
    display_string = now_time.strftime("%A %d %B, %Y | %H:%M:%S")

    cv2.putText(image, display_string, (x_print, y_print), cv2.FONT_HERSHEY_SIMPLEX, text_size, color_to_print, text_width)
    cv2.imshow(window_title, image)

    k = cv2.waitKey(1)
    if k == ord('q'): # wait for exit key
        cv2.destroyAllWindows()
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('output.png', image)
        cv2.destroyAllWindows()
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()