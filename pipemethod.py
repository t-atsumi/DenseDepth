import os
import errno
import time
import struct

def openFIFO(PATH_to_pipe,worr):
    #FIFO = "/tmp/ktl-fifo/torque"
    FIFO = PATH_to_pipe
    print(FIFO)
    try:
        os.mkfifo(FIFO)
    except OSError as oe:
        if oe.errno != errno.EEXIST:
            raise

    print("open PIPE...")
    if worr == "w":
        fifo = os.open(FIFO,os.O_RDWR|os.O_NONBLOCK)
    elif worr =="r":
        fifo = os.open(FIFO,os.O_RDONLY|os.O_NONBLOCK)
    else :
        fifo = os.open(FIFO,os.O_RDWR|os.O_NONBLOCK)
    print("FIFO opened")
    return fifo

def pack_and_write(data_,fifo,sleeptime=1.0):
    data_size = len(data_)
    packing_method = "<" + "d"*data_size
    os.write(fifo,struct.pack(packing_method,*data_))
    time.sleep(sleeptime)
    return 0

def read_and_unpack(fifo,sizeoflist):
        data = os.read(fifo,1024)
        try:
            unpack_method = "<" + "d"*sizeoflist
            out = struct.unpack(unpack_method,data)
        except:
            out = 0
        
        return out



def main():
    fifo = openFIFO("/tmp/ktl-fifo/torque","w")
    n=0
    while True:
        try:
            data = [1*n,2.000,0.03,4,5,6,7,8,9]
            pack_and_write(data,fifo)
            n+= 1

        except KeyboardInterrupt:
            print ('Finish')
            os.close(fifo)
            break

#python hoge.py と入力されてこのファイルが動くときだけmain()が動く、これがないとimportしただけでmain()が動く
if __name__ == "__main__":
    main()