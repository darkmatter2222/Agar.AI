import scripts.screen_interface as si
import scripts.game_interface as gi
import ctypes
import json
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pyautogui

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



# load model
model = tf.keras.models.load_model('..\\Models\\MultiClassV1.h5')

# Classes
classesOrigional = json.loads(open('..\\Models\\Classes.json').read())
classes = {}
for cl in classesOrigional:
    classes[classesOrigional[cl]] = cl

ratio_multi = 2
target_ratio = (192 * ratio_multi, 97 * ratio_multi)

# Starting Main Loop (will run faster if using Tensorflow + GPU)
print('Started')
while 1 == 1:
    # Grab Screen
    image = SI.get_screen_by_range(0, 80, 1920, 970)

    #image.thumbnail(target_ratio)
    img = image
    img = img.resize(target_ratio)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, target_ratio[0], target_ratio[1], 3)
    prediction = model.predict(img)
    x, y = GI.set_mouse_pos(np.argmax(prediction)* 10)
    print(f'Angle:{np.argmax(prediction)* 10} X:{x} Y:{y}')
    if keyboard.is_pressed('c') and prediction[0][np.argmax(prediction)] > 0.2:
        pyautogui.moveTo(x, y)


#plt.scatter(centerPoint[0], centerPoint[1], color='red')
#plt.scatter(centerPoint[0], centerPoint[1] + -200, color='green')
#plt.scatter(centerPoint[0] - 200, centerPoint[1] + -200, color='green')

#plt.scatter(x, y, color='orange')
#plt.show()