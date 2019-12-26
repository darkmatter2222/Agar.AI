import scripts.screen_interface as si
import scripts.game_interface as gi
import ctypes
import os
import keyboard
import uuid

GI = gi.GameInterface()

# find center of screen
user32 = ctypes.windll.user32
screenSize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
centerPoint = tuple(i/2 for i in screenSize)
print('Screen Size X:%d y:%d' % screenSize)
print('Targeting Center X:%d y:%d' % centerPoint)
GI = gi.GameInterface()
SI = si.ScreenInterface()
GI.center_x = centerPoint[0]
GI.center_y = centerPoint[1]
GI.range_classifications = 10

while True:
    angle = GI.get_mouse_class()
    print(angle)