import scripts.screen_interface as si
import scripts.game_interface as gi
import ctypes
import os
import keyboard
import uuid

# create directories
base_directory = 'N:\\Projects\\Agar.AI\\Training Data'
train_directory = '\\Train'
validate_directory = '\\Validation'
for d_name in range(0, 37):
    target_train_directory = f'{base_directory}{train_directory}\\{d_name}'
    target_validate_directory = f'{base_directory}{validate_directory}\\{d_name}'
    if not os.path.exists(target_train_directory):
        os.makedirs(target_train_directory)
    if not os.path.exists(target_validate_directory):
        os.makedirs(target_validate_directory)

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

# start collection
while True:
    if keyboard.is_pressed('c'):
        angle = GI.get_mouse_class()
        # 1080p screen, in chrome, bookmarks tab turned off.
        image = SI.get_screen_by_range(0, 80, 1920, 970)
        target_directory = f'{base_directory}{train_directory}\\{angle}\\'
        generatedGUID = str(uuid.uuid1())
        print(angle)
        image.save(target_directory + generatedGUID + '.png', 'png')

