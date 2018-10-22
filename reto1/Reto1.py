#Recognition of chocolates

import numpy as np
import cv2
from tkinter import *
from tkinter import ttk
import time

#Start    

print ("Bienvenido")

#Ventana raíz

def detection():
    
    tiempo_inicial=time.time() 
    tiempo_prueba=0
    cap = cv2.VideoCapture('prueba4.avi') #abre el archivo si es un video. escribir 0 para camara en vivo
    kernel = np.ones((55,55),np.uint8)
    background=np.zeros((480,380),np.uint8)
    feature = []
    onehistr=[]
    count=0
    tipo=0
    cy=0
    num=0
    etiquetas=[' ','Blanca','Mix','Negra','Roja','Azul']
    conteo=[0,0,0,0,0,0]
    while(cap.isOpened()):
        ret, frameoriginal = cap.read()
        if ret is True:
            frame=frameoriginal[:,120:500]
            median = cv2.medianBlur(frame,5)
            edges = cv2.Canny(median,120,240)
            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            masked_data = cv2.bitwise_and(frame,frame, mask=closing)
            img=np.array(masked_data)
            
        
            (_,contours,hierarchy)=cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for pic,contour in enumerate(contours):
                area=cv2.contourArea(contour)
                if (area>300):
                    M = cv2.moments(contour)
                    cy = int(M['m01']/M['m00'])
    
                    x,y,w,h=cv2.boundingRect(contour)
                    frameoriginal=cv2.rectangle(frameoriginal,(x+120,y),(x+w+120,y+h),(0,255,255),2)
                    cv2.putText(frameoriginal,str(etiquetas[tipo]),(x+100,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255))
                    
             
            tiempo_actual=time.time()
            if tiempo_actual-tiempo_prueba>2:
                tipo=0
            if 240<cy<250 and tiempo_actual-tiempo_prueba>2:
                tiempo_prueba=tiempo_actual
                num=num+1
                while count<20:
                                
                    count=count+1
                    
                    bhist = cv2.calcHist([img],[0],None,[256],[0,256])
                    b=np.float32(np.argmax(bhist[1:100,0],0)+1)
                
                    ghist = cv2.calcHist([img],[1],None,[256],[0,256])
                    g=np.float32(np.argmax(ghist[100:,0],0)+100)
                    
                    rhist = cv2.calcHist([img],[2],None,[256],[0,256])
                    r=np.float32(np.argmax(rhist[1:150,0],0)+1)
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hhist= cv2.calcHist([hsv], [0], None, [180], [0, 180])
                    h=np.float32(np.argmax(hhist[1:,0],0)+1) 
                    
                    shist= cv2.calcHist([hsv], [1], None,[256],[0,256])
                    s=np.float32(np.argmax(shist[1:,0],0)+1) 
                    
                    vhist= cv2.calcHist([hsv], [2], None,[256],[0,256])
                    v=np.float32(np.argmax(vhist[100:,0],0)+100) 
                
                    onehistr=[r,g,b,h,s,v]
        
                    
                    feature.append(onehistr)
                
                    #print(count)
                       
                #print(result)
                feature = np.squeeze(feature)
                svm = cv2.ml.SVM_load('svm_data.dat')
                result = svm.predict(feature)[1].ravel()
                result = list(map(int, result))
                tipo=(np.argmax(np.bincount(result)))
                feature=[]
                count=0
                conteo[tipo]=conteo[tipo]+1
                
                ja=StringVar()
                ja.set(conteo[5])
                jet=Entry(root,textvariable=ja)
                jet.place(x=170,y=115)
                labelconteojet=Label(root,text="Jet Azul")
                labelconteojet.place(x=30,y=115)
            
                b=StringVar()
                b.set(conteo[1])
                blanca=Entry(root,textvariable=b)
                blanca.place(x=170,y=230)
                labelconteoblanca=Label(root,text="Jumbo Blanca")
                labelconteoblanca.place(x=30,y=230)
            
                n=StringVar()
                n.set(conteo[3])
                negra=Entry(root,textvariable=n)
                negra.place(x=170,y=370)
                labelconteonegra=Label(root,text="Jumbo Negra")
                labelconteonegra.place(x=30,y=370)
            
                m=StringVar()
                m.set(conteo[2])
                mix=Entry(root,textvariable=m)
                mix.place(x=170,y=495)
                labelconteomix=Label(root,text="Jumbo Mix")
                labelconteomix.place(x=30,y=495)
          
                r=StringVar()
                r.set(conteo[4]) 
                roja=Entry(root,textvariable=r)
                roja.place(x=170,y=655)
                labelconteoroja=Label(root,text="Jumbo Maní")
                labelconteoroja.place(x=30,y=655)
            
            cv2.imshow('En vivo',frameoriginal)
            
            
             
    
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:

            print("Esperando nueva detección")
            nr_jet_azul= conteo[5]
            print ("número de jet azul detectadas",nr_jet_azul)
            nr_flow_negra=conteo[3]
            print ("número de flow negra detectadas",nr_flow_negra)
            nr_flow_blanca=conteo[1]
            print ("número de flow blanca detectadas",nr_flow_blanca)
            nr_jumbo_naranja=conteo[2]
            print ("número de jumbo naranja detectadas",nr_jumbo_naranja)
            nr_jumbo_roja=conteo[4]
            print ("número de jumbo roja detectadas",nr_jumbo_roja)
            
        
            
            time.sleep(3)  #Espera de 3 segund
            tiempo_final=time.time()
            tiempo_ejecucion=tiempo_final-tiempo_inicial
            tiempoentero=int(tiempo_ejecucion)
            print ("el tiempo de ejecucion fue",tiempoentero,"segundos")
           
            
            timerwrite=StringVar()
            timerwrite.set(tiempoentero)
            timewindow=Entry(root,textvariable=timerwrite)
            timewindow.place(x=540,y=630)
            
            break
        
    cap.release()
    cv2.destroyAllWindows()


#GUI
    
root=Tk()
root.title("Detección en la banda")
root.geometry("1450x8000")


#Images
jetazul=PhotoImage(file="jetazul.png")

labelyet=Label(root,image=jetazul)
labelyet.place(x=20,y=20)


flowblanca=PhotoImage(file="jumboflowblanca.png")

labelblanca=Label(root,image=flowblanca)
labelblanca.place(x=13,y=120)


flownegra=PhotoImage(file="jumbonegra.png")

labelnegra=Label(root,image=flownegra)
labelnegra.place(x=-12,y=235)


jumbomix=PhotoImage(file="jumbomix.png")

labelmix=Label(root,image=jumbomix)
labelmix.place(x=10,y=380)


jumboroja=PhotoImage(file="jumborojar.png")

labelroja=Label(root,image=jumboroja)
labelroja.place(x=-15,y=500)

boton_start = Button(root,text = "Start", fg="blue")
boton_start.place(x=30,y=10)
boton_start.config(command=detection)


label1= Label(root, text = "Presiona start para comenzar la detección", fg="gray")
label1.place(x=100,y=10)


labeltime=Label(root, text = "El tiempo de ejecución en segundos fue:", fg="gray")
labeltime.place(x=500,y=600)



root.mainloop()


            

