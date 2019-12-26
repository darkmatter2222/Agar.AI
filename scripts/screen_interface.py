import pyautogui


class ScreenInterface:
    def __init__(self):
        print("in init")

    def get_screen_by_range(self, left, top, width, height):
        return pyautogui.screenshot(region=(left, top, width, height))

