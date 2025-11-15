import cv2
import math

def construct_loomis_sphere(image, center, radius, angle):
    cx, cy = int(center[0]), int(center[1])

    
    cv2.circle(
        image,
        (cx, cy),
        int(radius),
        (0, 255, 0),
        2
    )

    rad = math.radians(angle)

    plane_x = int(cx + radius * 0.45 * math.cos(rad))
    plane_y = int(cy + radius * 0.45 * math.sin(rad))

    ellipse_w = int(radius * 0.45)
    ellipse_h = int(radius * 0.70)

    cv2.ellipse(
        image,
        (plane_x, plane_y),
        (ellipse_w, ellipse_h),
        angle,      
        0, 360,
        (0, 255, 0),
        2
    )

    return image
