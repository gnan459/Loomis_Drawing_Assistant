import cv2
import math

def construct_loomis_sphere(image, center, radius, direction, face_landmarks):
    cx, cy = int(center[0]), int(center[1])

    cv2.circle(image, (cx, cy), int(radius), (0,255,0), 1)

    nose_x = face_landmarks[1][0]

    # Face center (average of all x coordinates)
    xs = [p[0] for p in face_landmarks]
    face_center_x = sum(xs) / len(xs)

    # If nose is extremely close to the face center → straight face → no ellipse
    if abs(nose_x - face_center_x) < 5:
        return image

    offset = int(radius * 0.40)

    # place ellipse on correct side
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

    # Chin landmark
    chin = face_landmarks[152]

    # Compute midpoint between both eyes
    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)

    # Chin point
    chin_x = int(chin[0])
    chin_y = int(chin[1])

    # Draw the line (from eyes → chin)
    cv2.line(image, (eye_center_x, eye_center_y), (chin_x, chin_y), (0,255,0), 1)

    return image

def construct_brow_line(image, face_landmarks):
    left_brow = face_landmarks[105]
    right_brow = face_landmarks[334]

    x1, y1 = int(left_brow[0]), int(left_brow[1])
    x2, y2 = int(right_brow[0]), int(right_brow[1])

    cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 1)

    return image


