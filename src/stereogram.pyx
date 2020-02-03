import random
import numpy as np



DPI = 72
E = round(2.5*DPI)
MU = 1./3.


def separation(z):
    return int(round((1.-(MU*z)) * (E/(2-(MU*z)))))


FAR = separation(0)

DOTS_Y = .90

def get_stereogram(depth_map):
    MAX_X = len(depth_map)
    MAX_Y = len(depth_map[0])

    output_map = np.ndarray((MAX_X, MAX_Y))

    for y in range(MAX_Y):
        row = [x for x in range(MAX_X)]
        row_same = [x for x in range(MAX_X)]

        for x in range(MAX_X):
            row_same[x] = x

        for x in range(MAX_X):
            stereo_separation = separation(depth_map[x][y])
            left = x - int(stereo_separation / 2)
            right = left + stereo_separation
            if 0 <= left and right < MAX_X:
                visible = False
                t = 1
                zt = 0
                while True:
                    zt = depth_map[x][y] + (2*(2 - (MU*depth_map[x][y]*t/(MU*E))))
                    visible = depth_map[x-t][y] < zt and depth_map[x+t][y] < zt
                    t += 1
                    if not (visible and zt < 1):
                        break
                if visible:
                    l = row_same[left]
                    while l != left and l != right:
                        if l < right:
                            left = l
                            l = row_same[left]
                        else:
                            row_same[left] = right
                            left = right
                            l = row_same[left]
                            right = l
                    row_same[left] = right
        for x in range(MAX_X-1, 0, -1):
            if row_same[x] == x:
                row[x] = random.randint(0, 1)
            else:
                row[x] = row[row_same[x]]
            output_map[x, y] = row[x]

    circle_1_center = (int(MAX_Y * DOTS_Y), int((MAX_X/2) - (FAR/2)))
    circle_2_center = (int(MAX_Y * DOTS_Y), int((MAX_X/2) + (FAR/2)))

    return output_map, circle_1_center, circle_2_center











