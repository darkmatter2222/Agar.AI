import pyautogui


def get_screen_by_range(left, top, width, height):
    return pyautogui.screenshot(region=(left, top, width, height))

