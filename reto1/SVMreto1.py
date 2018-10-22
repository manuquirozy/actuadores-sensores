import numpy as np
import cv2
import glob
import pandas as pd
from matplotlib import pyplot as plt

kernel = np.ones((55,55),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
def training_data(chocolatinas,labels,directory,tag):
    files = glob.glob (directory)
    for myFile in files:
        frame = cv2.imread (myFile)
        frame=frame[:,120:500]
        median = cv2.medianBlur(frame,5)
        edges = cv2.Canny(median,120,240)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        masked_data = cv2.bitwise_and(frame,frame, mask=closing)
        chocolatinas.append(masked_data)
        etiquetas.append (int(tag))
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

training_data(chocolatinas,etiquetas,fondodir,0)
training_data(chocolatinas,etiquetas,blancadir,1)
training_data(chocolatinas,etiquetas,mixdir,2)
training_data(chocolatinas,etiquetas,negradir,3)
training_data(chocolatinas,etiquetas,rojadir,4)
training_data(chocolatinas,etiquetas,azuldir,5)

print(np.array(chocolatinas).shape[0], " imagenes recuperadas")

chocolatinas=np.array(chocolatinas)
etiquetas=np.array(etiquetas)

rand = np.random.RandomState(10)
shuffle = np.random.permutation(len(chocolatinas))
chocolatinas= chocolatinas[shuffle]
etiquetas =  etiquetas[shuffle]

print('Calculando atributos ... ')

feature = []
onehistr=[]
for img in chocolatinas:
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
feature = np.squeeze(feature)

print('Separando datos entre entrenamiento (90%) y prueba (10%)... ')
train_n=int(0.9*len(feature))
chocolatinas_train, chocolatinas_test = np.split(chocolatinas, [train_n])
feature_train, feature_test = np.split(feature, [train_n])
etiquetas_train, etiquetas_test = np.split(etiquetas, [train_n])

print('SVM... ')
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.trainAuto(feature_train, cv2.ml.ROW_SAMPLE, etiquetas_train)
svm.save('svm_data.dat')

result = svm.predict(feature_test)[1].ravel()
accuracy = (etiquetas_test == result).mean()
print('Porcentaje de exactitud: %.2f %%' % (accuracy*100))
y_actu = pd.Series(etiquetas_test, name='Real')
y_pred = pd.Series(result, name='Estimado por SVM')
df_confusion = pd.crosstab(y_actu,y_pred)
plot_confusion_matrix(df_confusion)
print('Matriz de confusión:')
print('0: Fondo, 1: Blanca, 2: Mix, 3: Negra, 4: Roja, 5: Azul')