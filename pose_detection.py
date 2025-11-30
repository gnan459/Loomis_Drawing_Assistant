import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence=0.5
)

pose_img = mp_pose.Pose(static_image_mode=True, 
                    min_detection_confidence=0.5, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(input_img, results, results_face,
        landmarks_c=(234, 63, 247), connection_c=(117, 249, 77), thickness=1, circle_r=1):
    
    height, width, _ = input_img.shape

    face_landmarks = []

    if results_face.multi_face_landmarks:
        for lm in results_face.multi_face_landmarks[0].landmark:
            face_landmarks.append((int(lm.x * width), int(lm.y * height), lm.z))
            cv2.circle(input_img, (int(lm.x * width), int(lm.y * height)), 1, (0,255,0), -1)

    pose_landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            input_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=connection_c, thickness=thickness, circle_radius=circle_r),
            mp_drawing.DrawingSpec(color=landmarks_c, thickness=thickness, circle_radius=circle_r),
        )
        for landmark in results.pose_landmarks.landmark:
            pose_landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z * width))

    return pose_landmarks, face_landmarks

# def displayLandmarks(original_img, output_img):
#     plt.figure(figsize=(14, 7))

#     # Original image
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_img[:, :, ::-1])
#     plt.title("Original Image")
#     plt.axis("off")

#     # Pose + Face Mesh overlay image
#     plt.subplot(1, 2, 2)
#     plt.imshow(output_img[:, :, ::-1])
#     plt.title("Pose + Face Landmarks")
#     plt.axis("off")

#     plt.show()

def detect_pose_and_face(input_file, display=True):
    if isinstance(input_file, str):
        image = cv2.imread(input_file)
    else:
        image = input_file

    if image is None:
        raise ValueError("Image not found or unable to read.")

    output_img = image.copy()

    RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    results = pose_img.process(RGB_img)
    results_face = face_mesh.process(RGB_img)

    pose_landmarks, face_landmarks = draw_landmarks(input_img=output_img, results=results, results_face=results_face)

    

    # if display:
    #     displayLandmarks(image, output_img)

    # print("Face Lanmdarks: ", face_landmarks)
    # print("Pose Landmarks: ", pose_landmarks)

    return {
    "image": output_img,
    "face": face_landmarks,
    "pose": pose_landmarks
    }




#Testing

# data = detect_pose_and_face("D:/Loomis_Drawing_Assistant/tests/image_2.jpg", display=True)
# from geometry_utils import calculate_head_dimensions
# head, width, center = calculate_head_dimensions(data['face'])
# print(head, width, center)


# data = detect_pose_and_face("D:/Loomis_Drawing_Assistant/tests/image_4.jpg")
# from geometry_utils import calculate_head_dimensions, compute_centerline, compute_face_turn_angle
# from render_steps import construct_loomis_sphere, construct_vertical_line, construct_brow_line, construct_nose_line, construct_chin_line, construct_ellipse_vertical_line, construct_jaw_line, construct_outer_face_line

# radius, center, skull_top = calculate_head_dimensions(data['face'])
# angle, nose, chin = compute_centerline(data['face'])
# direction = compute_face_turn_angle(data["face"])
# image = cv2.imread("D:/Loomis_Drawing_Assistant/tests/image_4.jpg")
# image= construct_loomis_sphere(image, center, radius, direction, data["face"])
# image = construct_vertical_line(image, data["face"])
# image = construct_brow_line(image, data["face"])
# image = construct_nose_line(image, data["face"])
# image = construct_chin_line(image, data["face"])
# image = construct_ellipse_vertical_line(image,center,radius,direction, data["face"])
# image = construct_jaw_line(image, data["face"])
# image = construct_outer_face_line(image, data["face"], direction)

# cv2.imshow("Image", image)
# cv2.waitKey(0)
