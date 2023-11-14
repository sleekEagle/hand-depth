import serial
import struct
from datetime import datetime
from pathlib import Path
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shoot photos remotely with canon camera.')
    parser.add_argument('--outpath', type=str,
                        default='C:\\Users\\lahir\\data\\CPR_experiment\\test\\arduino\\',
                        help='where is the data stored')
    args = parser.parse_args()

    today =  datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    outfile=Path(args.outpath)/(str(today)+".txt")

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