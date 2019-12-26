import pyautogui
import math


class GameInterface:
    center_x = 0
    center_y = 0

    def __init__(self):
        print("in init")

    def get_mouse_angle(self):
        a = (self.center_x, self.center_y + -200)
        b = (self.center_x, self.center_y)
        c = self.get_mouse_pos()

        angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        return angle + 360 if angle < 0 else angle

    def get_mouse_pos(self):
        return pyautogui.position()

    def set_mouse_pos(self):
        return 0
