import serial
import struct
from datetime import datetime
from pathlib import Path

outpath='C:\\Users\\lahir\\data\\depth_sensor\\'
today =  datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
outfile=Path(outpath)/(str(today)+".txt")

ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM3'
ser.encoding='utf-8'
ser.open()
try:
    with open(outfile, 'w') as f:
        while(True):
                #read a line from the serial port
                l=ser.readline().decode('utf-8')
                num=l.replace("\r","").replace("\n","")
                #get current local time
                current_time = datetime.now()
                out_str=str(current_time)+" "+str(num)
                #write timestamp and value to file
                f.write(out_str)
                f.write('\n')
                print(out_str)
except KeyboardInterrupt:
    print('interrupted!')
finally:
    ser.close()