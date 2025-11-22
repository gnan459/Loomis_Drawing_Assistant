import cv2
import math

def construct_loomis_sphere(image, center, radius, direction, face_landmarks):
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(image, (cx, cy), int(radius), (0,255,0), 1)

    nose_x = face_landmarks[1][0]

    xs = [p[0] for p in face_landmarks]
    face_center_x = sum(xs) / len(xs)

    if abs(nose_x - face_center_x) < 5:
        return image

    offset = int(radius * 0.40)

    plane_x = cx + offset if direction == "right" else cx - offset
    plane_y = cy

    ellipse_w = int(radius * 0.45)
    ellipse_h = int(radius * 0.70)

    cv2.ellipse(
        image,
        (plane_x, plane_y),
        (ellipse_w, ellipse_h),
        0, 0, 360,
        (0,255,0), 1
    )

    return image



def construct_vertical_line(image, face_landmarks):
    left_eye  = face_landmarks[105]
    right_eye = face_landmarks[334]

    chin = face_landmarks[152]

    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)

    chin_x = int(chin[0])
    chin_y = int(chin[1])

    cv2.line(image, (eye_center_x, eye_center_y), (chin_x, chin_y), (0,255,0), 1)

    return image

def construct_brow_line(image, face_landmarks):
    left_brow = face_landmarks[105]
    right_brow = face_landmarks[334]

    x1, y1 = int(left_brow[0]), int(left_brow[1])
    x2, y2 = int(right_brow[0]), int(right_brow[1])

    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)

    return image

def construct_nose_line(image, face_landmarks):
    left_nostril = face_landmarks[98]
    right_nostril = face_landmarks[327]

    x1, y1 = int(left_nostril[0]), int(left_nostril[1])
    x2, y2 = int(right_nostril[0]), int(right_nostril[1])

    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)

    return image

def construct_chin_line(image, face_landmarks):
    nx, ny = int(face_landmarks[152][0]), int(face_landmarks[152][1])

    line_len = 20  

    x1 = nx - line_len    
    y1 = ny

    x2 = nx + line_len    
    y2 = ny

    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)

    return image

def construct_ellipse_vertical_line(image, center, radius, direction, face_landmarks):

    nose_x = face_landmarks[1][0]

    xs = [p[0] for p in face_landmarks]
    face_center_x = sum(xs) / len(xs)

    if abs(nose_x - face_center_x) < 5:
        return image
    
    cx, cy = int(center[0]), int(center[1])

    offset = int(radius * 0.40)
    ex = cx + offset if direction == "right" else cx - offset
    ey = cy

    line_len = int(radius * 1.2)
    x1, y1 = ex, ey
    x2, y2 = ex, ey + line_len

    cv2.line(image, (x1,y1), (x2, y2), (0,255,0), 1)
    return image

def construct_jaw_line(image, face_landmarks):
    chin = face_landmarks[152]
    left_jaw = face_landmarks[172]
    right_jaw = face_landmarks[397]

    chin = (int(chin[0]), int(chin[1]))
    left_jaw = (int(left_jaw[0]), int(left_jaw[1]))
    right_jaw = (int(right_jaw[0]), int(right_jaw[1]))

    cv2.line(image, left_jaw, chin, (0,255,0), 1)

    cv2.line(image, right_jaw, chin, (0,255,0), 1)

    return image

def construct_outer_face_line(image, face_landmarks, direction):
    chin = face_landmarks[152]

    if direction == "right":
        eye = face_landmarks[33]
        jaw = face_landmarks[172]
    else:
        eye = face_landmarks[263]
        jaw = face_landmarks[397]

    eye = int(eye[0]), int(eye[1])
    jaw = int(jaw[0]), int(jaw[1])
    chin = int(chin[0]), int(chin[1])

    cv2.line(image, eye, jaw, (0,255,0), 1)
    cv2.line(image, jaw, chin, (0,255,0), 1)

    return image

