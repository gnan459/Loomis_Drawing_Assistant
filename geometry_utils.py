import math

def calculate_head_dimensions(face_landmarks):

    if face_landmarks is None or len(face_landmarks) < 468:
        raise ValueError("Face landmark data is incomplete")

    xs = [p[0] for p in face_landmarks]
    ys = [p[1] for p in face_landmarks]

    left   = min(xs)
    right  = max(xs)
    top    = min(ys)
    bottom = max(ys)

    face_width  = right - left
    face_height = bottom - top

    radius = int(face_height * 0.55)

    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)
    center = (center_x, center_y)

    skull_top = (center_x, center_y - radius)

    return radius, center, skull_top




def compute_centerline(face_landmarks):
    if face_landmarks is None or len(face_landmarks) < 468:
        raise ValueError("Face landmark data is incomplete")
    
    nose = face_landmarks[1]
    chin = face_landmarks[152]

    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]

    
    angle = math.degrees(math.atan2(dy, dx))

    
    angle = angle + 90

    return angle, (nose[0], nose[1]), (chin[0], chin[1])



