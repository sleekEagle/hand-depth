import pyautogui
from win32gui import FindWindow, GetWindowRect
import win32gui
import time
import argparse

def winEnumHandler( hwnd,nimgs):
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
            #continuous shooting
            if nimgs==-1:
                pyautogui.mouseDown(button='left')
            else:
                for n in range(nimgs):
                    pyautogui.click(button='left')
                    time.sleep(1)

            # time.sleep(1)
            # pyautogui.mouseUp(button='left')
            # print(x,y)
'''
n_imgs: number of photos to take 
-1 : continuous shooting. click somewhere else to end.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shoot photos remotely with canon camera.')
    parser.add_argument('--n', type=int,
                        default=1,
                        help='nuber of images to shoot. -1 for continuous shooting.')
    args = parser.parse_args()
    win32gui.EnumWindows( winEnumHandler,args.n)
