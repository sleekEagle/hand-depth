import pyautogui
from win32gui import FindWindow, GetWindowRect
import win32gui
import time

def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        name= win32gui.GetWindowText( hwnd )
        if 'T100' in name:
            print('found!')
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1]
            click_x=x+255
            click_y=y+99
            pyautogui.moveTo(x=click_x,y=click_y) 
            pyautogui.mouseDown(button='left')
            # time.sleep(1)
            # pyautogui.mouseUp(button='left')
            # print(x,y)

win32gui.EnumWindows( winEnumHandler, None )
