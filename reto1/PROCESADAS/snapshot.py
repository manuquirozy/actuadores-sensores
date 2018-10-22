import cv2
import numpy as np
import glob

kernel = np.ones((55,55),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

def training_data(chocolatinas,labels,directory,tag, name):
    files = glob.glob (directory)
    c=0
    for myFile in files:
        c=c+1
        frame = cv2.imread (myFile)
        frame=frame[:,120:500]
        median = cv2.medianBlur(frame,5)
        edges = cv2.Canny(median,120,240)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        erosion = cv2.erode(closing,kernel2,iterations = 5)
        masked_data = cv2.bitwise_and(frame,frame, mask=erosion)
        chocolatinas.append(masked_data)
        etiquetas.append (int(tag))
        cv2.imwrite(str(name)+str(c)+".jpg", masked_data)
    return chocolatinas

print("Recuperando imagenes de entrenamiento...")    


fondodir="C:/Users/mquiroz/Desktop/reto/FONDO/*.jpg" #FONDO:0
blancadir="C:/Users/mquiroz/Desktop/reto/JUMBO BLANCA/*.jpg" #BLANCA: 1
mixdir="C:/Users/mquiroz/Desktop/reto/JUMBO MIX/*.jpg" #MIX: 2
negradir="C:/Users/mquiroz/Desktop/reto/JUMBO NEGRA/*.jpg" #NEGRA: 3
rojadir="C:/Users/mquiroz/Desktop/reto/JUMBO ROJA/*.jpg" #ROJA: 4
azuldir="C:/Users/mquiroz/Desktop/reto/AZUL/*.jpg" #AZUL: 5

chocolatinas= []
etiquetas=[]
nombres=[' ','Blanca','Mix','Negra','Roja','Azul']

training_data(chocolatinas,etiquetas,fondodir,0, nombres[0])
training_data(chocolatinas,etiquetas,blancadir,1, nombres[1])
training_data(chocolatinas,etiquetas,mixdir,2, nombres[2])
training_data(chocolatinas,etiquetas,negradir,3, nombres[3])
training_data(chocolatinas,etiquetas,rojadir,4, nombres[4])
training_data(chocolatinas,etiquetas,azuldir,5, nombres[5])


    






