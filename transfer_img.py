
#方法可行，等待批量製作
import glob
from importlib.resources import path
from PIL import Image

open_file_path=str(input("enter the folder which you put your untransfered image\n"))
open_file_path=open_file_path+'/*'
final_file_path=str(input("enter the folder which you put your transfered image\n"))


file=glob.glob(open_file_path) 
num=1

for i in file:
    im=Image.open(i)
    bg = Image.new("RGB", im.size, (255,255,255))
    bg.paste(im,im)
    bg.resize((28,28))
    numa='B_'+str(num)+'.jpg'
    bg.save(f"{final_file_path}/{numa}")
    print(numa,"saved")
    num+=1