from tkinter import * 
import cv2
import numpy as np 
import os
import math
import matplotlib.pyplot as plt 
import pandas as pd

from PIL import Image, ImageTk 
from tkinter import ttk
from tkinter.filedialog import askopenfilename 
from skimage import morphology
from skimage.restoration import denoise_nl_means, estimate_sigma 
from skimage import img_as_float
from scipy import ndimage as nd
from skimage.filters import threshold_multiotsu
from skimage import measure, io, img_as_ubyte, data 
from os import remove
from skimage.measure import label, regionprops 

count = 0

class Window:
    def _init_(self, window): 
        self.window = window
        self.window.geometry("1250x550") 
        self.window.configure(bg="black") 
        self.Botones()
        #Botones de la pantalla principal def Botones(self):
        self.window.title("Osteosarcoma Lab")

        self.label = Label(self.window, text = "OSTEOSARCOMA LAB", bg= "black", fg = "white", font=("Roboto Cn",40))
        self.label.pack()

        self.label = Label(self.window, text = "Seleccione una opcion", bg= "black", fg
        = "white", font=("Roboto Cn",10)) 
        self.label.pack()

        # Procesamiento Rayos X

        self.b1 = Button(self.window, text=" Rayos X ", height = 5, width = 12 ,bg= "black", fg = "white",font=("Roboto Cn",10), command = self.rayosx)
        self.b1.place(x=120, y=140)
        
        # Procesamiento en Resonancia Magnetica

        self.b3 = Button(self.window, text="MRI", height = 5, width = 12 , bg= "black", fg = "white", font=("Roboto Cn",10), command = self.resonm)
        self.b3.place(x=120, y=270)

        self.bfin = Button(self.window, text="Cerrar", height = 5, width = 28, bg
        ="gray7", fg ="gray92" ,font=("Roboto Cn",10), command = self.window.quit) 
        self.bfin.place(x=60, y=400)


    def rayosx(self):
        filename = askopenfilename() 
        image = cv2.imread(filename) 
        plt.imsave("original.jpg",image) 
        img = img_as_float(image)
        sigma_est = np.mean(estimate_sigma(img, multichannel=True)) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode = True, patch_size=5, patch_distance=3, multichannel=True)
        plt.imsave("clean.jpg",denoise_img, cmap = "gray") 
    
    def gammaCorrection(src, gamma):
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)] 
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)
    
    img2 = cv2.imread('clean.jpg')
    gammaImg = gammaCorrection(img2, 0.4)
    gray = cv2.cvtColor(gammaImg, cv2.COLOR_BGR2GRAY) 
    filterSize =(3, 3)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8)) 
    claheNorm = clahe.apply(gray)

    def pixelVal(pix, r1, s1, r2, s2): 
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix 
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
        
    r1 = 50
    s1 = 0
    r2 = 200
    s2 = 255
    pixelVal_vec = np.vectorize(pixelVal)
    cs = pixelVal_vec(claheNorm, r1, s1, r2, s2) 
    plt.imsave("cont.jpg",cs, cmap="gray")

    def gammaCorrection(src, gamma): 
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)] 
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table)

    img21 = cv2.imread('cont.jpg')
    gammaImg2 = gammaCorrection(img21, 0.9) 
    plt.imsave('corr.jpg', gammaImg2)
    self.label = Label(self.window, text = "Su imagen esta lista!", bg= "black", fg = "white")
    self.label.place(x=800, y=470)

    kernel1 = np.array([[0, -1, 0],
    [-1, 5,-1],
    [0, -1, 0]])
    image_sharp = cv2.filter2D(src=gammaImg2, ddepth=-1, kernel=kernel1) 
    plt.imsave('Imagen final.jpg', image_sharp)

    image1 = Image.open("Imagen final.jpg").resize((330,330)) 
    test = ImageTk.PhotoImage(image1)
    label1 = Label(image=test) 
    label1.image = test
    image2 = Image.open("original.jpg").resize((330,330)) 
    test2 = ImageTk.PhotoImage(image2)
    label2= Label(image=test2) 
    label2.image = test2

    label1.place(x=500, y=120) 
    label2.place(x=850, y=120)
    remove("corr.jpg") 
    remove("cont.jpg") 
    remove("clean.jpg")


    def resonm(self):
        filename1 = askopenfilename() 
        imagerm = cv2.imread(filename1)
        plt.imsave("originalrm.jpg",imagerm,cmap="gray") 
        kernel1 = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
        image_sharp = cv2.filter2D(src=imagerm, ddepth=-1, kernel=kernel1) 
        plt.imsave('sharp.jpg', image_sharp)

    def gammaCorrection(src, gamma):
        invGamma = 1 / gamma
        table = [((i / 255) ** invGamma) * 255 for i in range(256)] 
        table = np.array(table, np.uint8)
        return cv2.LUT(src, table) 
    
    imggam = image_sharp.copy()
    gammaImg = gammaCorrection(imggam, 0.8)

    thresholds = threshold_multiotsu(gammaImg, classes=5) 
    regions = np.digitize(gammaImg, bins=thresholds)
    imgthr = cv2.normalize(regions, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    kernel = np.ones((3,3),np.uint8)

    kmean = cv2.cvtColor(imgthr, cv2.COLOR_BGR2RGB) 
    pixel_values = kmean.reshape((-1, 3))
    pixel_values = np.float32(pixel_values) 
    criteria = (cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 6
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers) 
    labels = labels.flatten()
    segmimg = centers[labels.flatten()] 
    segmimg = segmimg.reshape(kmean.shape)
    maskimg1 = np.copy(segmimg) 
    maskimg1 = maskimg1.reshape((-1, 3)) 
    cluster = 1
    maskimg1[labels == cluster] = [255, 0, 0] 
    maskimg1 = maskimg1.reshape(segmimg.shape) 
    plt.imsave("mask1.jpg", maskimg1, cmap="gray")
    maskimg2 = np.copy(segmimg) 
    maskimg2 = maskimg2.reshape((-1, 3)) 
    cluster = 2
    maskimg2[labels == cluster] = [0, 0, 255] 
    maskimg2 = maskimg2.reshape(segmimg.shape) 
    plt.imsave("mask2.jpg", maskimg2, cmap="gray")

    maskimg3 = np.copy(segmimg) 
    maskimg3 = maskimg3.reshape((-1, 3)) 
    cluster = 3
    maskimg3[labels == cluster] = [0, 255, 0] 
    maskimg3 = maskimg3.reshape(segmimg.shape) 
    plt.imsave("mask3.jpg", maskimg3, cmap="gray")

    maskimg4 = np.copy(segmimg) 
    maskimg4 = maskimg4.reshape((-1, 3))
    
    cluster = 4
    maskimg4[labels == cluster] = [255, 255, 0] 
    maskimg4 = maskimg4.reshape(segmimg.shape) 
    plt.imsave("mask4.jpg", maskimg4, cmap="gray")

    remove("sharp.jpg")

    img2 = cv2.imread('mask1.jpg')
    mask1 = cv2.inRange(img2, (0, 0, 50), (50, 50,255))
    apertura = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imsave("maskred1.jpg", apertura, cmap="gray") 
    img2 = cv2.imread("maskred1.jpg")
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    ret2,th2 =  cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    all_props = measure.regionprops(th2)
    props1 = measure.regionprops_table(th2, properties=['perimeter','label', 'area', 'equivalent_diameter',"filled_area"])

    img3 = cv2.imread('mask2.jpg')
    mask2 = cv2.inRange(img3, (50, 0, 0), (255, 50, 50))
    apertura2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imsave("maskblue1.jpg", apertura2, cmap="gray")
    img3 = cv2.imread("maskblue1.jpg")
    gray1 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) 
    ret3,th3 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    all_props1 = measure.regionprops(th3)
    props2 = measure.regionprops_table(th3, properties=['perimeter','label', 'area', 'equivalent_diameter',"filled_area"])
    escala = 32.14
    df = pd.DataFrame(props1) 
    a = df['area']
    ar= a/escala
    c = ar / math.pi 
    b = np.sqrt(c)

    df = pd.DataFrame(props2) 
    a1 = df['area']
    
    ar1 = a1/escala 
    c1 = ar1 / math.pi 
    b1 = np.sqrt(c1)
    areaeval= abs(b-b1) 
    liminf = 0.76
    limsup = 2
    value = float(areaeval)

    if value < liminf and value > 0 : 
        form = ((1.0519*b)-0.016)*2
        form=int(form)
        edge = cv2.Canny(apertura, 20, 150) 
        plt.imsave("canny.jpg", edge)
        ans = []
        a = int(form)
        img2 = cv2.imread("canny.jpg") 
        for y in range(0, edge.shape[0]):
            for x in range(0, edge.shape[1]): 
                if edge[y, x] != 0:
                    ans = ans + [[x, y]] 
                    imgWC = cv2.circle(img2,(x,y),form,(0,0,255),-1)
        plt.imsave("reseccion.jpg", imgWC, cmap="gray") 
        img3 = cv2.imread("reseccion.jpg")
        img4 = cv2.imread("axial.jpeg")
        dst = cv2.addWeighted(img4, 0.7, img3, 0.3, 0) 
        plt.imsave("final.jpg", dst, cmap="gray")
        image1 = Image.open("final.jpg").resize((330,330)) 
        test = ImageTk.PhotoImage(image1)
        label1 = Label(image=test) 
        label1.image = test 
        label1.place(x=500, y=100)

    image2 = Image.open("axial.jpeg").resize((180,180)) 
    test1 = ImageTk.PhotoImage(image2)
    label2 = Label(image=test1) 
    label2.image = test1 
    label2.place(x=830, y=80)
    image3 = Image.open("mask1.jpg").resize((180,180)) 
    test2 = ImageTk.PhotoImage(image3)
    label3 = Label(image=test2) 
    label3.image = test2 
    label3.place(x=1030, y=80)

    image4 = Image.open("mask2.jpg").resize((180,180)) 
    test3 = ImageTk.PhotoImage(image4)
    label4 = Label(image=test3) 
    label4.image = test3 
    label4.place(x=830, y=330)
    
    image5 = Image.open("mask3.jpg").resize((180,180)) 
    test4 = ImageTk.PhotoImage(image5)
    label5 = Label(image=test4) 
    label5.image = test4 
    label5.place(x=1030, y=330)

    self.label = Label(self.window, text = "Estadificaciòn IIB", bg= "black", fg = "white")
    self.label.place(x=980, y=50)

    self.label1 = Label(self.window, text = "Estadificaciòn IIB", bg= "black", fg = "white")
    self.label1.place(x=980, y=50)

    self.label2 = Label(self.window, text = "Estadificaciòn IIB", bg= "black", fg = "white")
    self.label2.place(x=980, y=50)

    self.label3 = Label(self.window, text = "Estadificaciòn IIB", bg= "black", fg = "white")
    self.label3.place(x=980, y=50)

    elif liminf >= value and value <= limsup : 
        form = ( (0.9485*b1) + 2.3546)
    form=int(form)
    edge = cv2.Canny(apertura2, 20, 150) 
    plt.imsave("canny.jpg", edge)
    ans = []
    a = int(form)
    img2 = cv2.imread("canny.jpg") 
    for y in range(0, edge.shape[0]):
        for x in range(0, edge.shape[1]): 
            if edge[y, x] != 0:
                ans = ans + [[x, y]] 
    imgWC = cv2.circle(img2,(x,y),form,(0,0,255),-1)
    plt.imsave("reseccion.jpg", imgWC, cmap="gray")
    img3 = cv2.imread("reseccion.jpg") 
    img4 = cv2.imread("axial.jpeg")
    dst = cv2.addWeighted(img4, 0.7, img3, 0.3, 0) 
    plt.imsave("final.jpg", dst, cmap="gray")
    image1 = Image.open("final.jpg").resize((330,330)) 
    test = ImageTk.PhotoImage(image1)
    label1 = Label(image=test) 
    label1.image = test 
    label1.place(x=500, y=100)

    image2 = Image.open("axial.jpeg").resize((180,180))
    
    test1 = ImageTk.PhotoImage(image2) 
    label2 = Label(image=test1) 
    label2.image = test1 
    label2.place(x=830, y=100)

    image3 = Image.open("mask1.jpg").resize((180,180)) 
    test2 = ImageTk.PhotoImage(image3)
    label3 = Label(image=test2) 
    label3.image = test2 
    label3.place(x=1030, y=100)

    image4 = Image.open("mask2.jpg").resize((180,180)) 
    test3 = ImageTk.PhotoImage(image4)
    label4 = Label(image=test3) 
    label4.image = test3 
    label4.place(x=830, y=300)

    image5 = Image.open("mask3.jpg").resize((180,180))
    test4 = ImageTk.PhotoImage(image5)
    label5 = Label(image=test4) 
    label5.image = test4
    label5.place(x=1030, y=300)

    self.label = Label(self.window, text = "Estadificaciòn IB", bg= "black", fg = "white")
    self.label.place(x=980, y=40)
    elif value > limsup:
    form = ((0.9222 * b1) + 1.3106)
    form=int(form)
    edge = cv2.Canny(apertura2, 10, 100) 
    plt.imsave("canny.jpg", edge)
    ans = []
    a = int(form)
    img2 = cv2.imread("maskblue1.jpg") 
    for y in range(0, edge.shape[0]):
    for x in range(0, edge.shape[1]): 
        if edge[y, x] != 0:
    ans = ans + [[x, y]] 
    imgWC = cv2.circle(img2,(x,y),form,(0,0,255),-1)
    plt.imsave("reseccion.jpg", imgWC, cmap="gray") 
    img3 = cv2.imread("reseccion.jpg")
    img4 = cv2.imread("axial.jpeg")
    dst = cv2.addWeighted(img4, 0.7, img3, 0.3, 0) 
    plt.imsave("final.jpg", dst, cmap="gray")
    image1 = Image.open("final.jpg").resize((330,330)) 
    test = ImageTk.PhotoImage(image1)
    label1 = Label(image=test) 
    label1.image = test
    
    label1.place(x=460, y=100)

    image2 = Image.open("axial.jpeg").resize((180,180)) 
    test1 = ImageTk.PhotoImage(image2)
    label2 = Label(image=test1) 
    label2.image = test1 
    label2.place(x=830, y=110)

    image3 = Image.open("mask1.jpg").resize((180,180)) 
    test2 = ImageTk.PhotoImage(image3)
    label3 = Label(image=test2) 
    label3.image = test2 
    label3.place(x=1030, y=110)

    image4 = Image.open("mask2.jpg").resize((180,180)) 
    test3 = ImageTk.PhotoImage(image4)
    label4 = Label(image=test3) 
    label4.image = test3 
    label4.place(x=830, y=335)

    image5 = Image.open("mask3.jpg").resize((180,180)) 
    test4 = ImageTk.PhotoImage(image5)
    label5 = Label(image=test4) 
    label5.image = test4 
    label5.place(x=1030, y=335)

    self.label = Label(self.window, text = "Estadificaciòn IIA", bg= "black", fg = "white")
    self.label.place(x=570, y=460)

    self.label1 = Label(self.window, text = "Imagen Original", bg= "black", fg = "white")
    self.label1.place(x=870, y=78)

    self.label2 = Label(self.window, text = "Lesiòn òsea", bg= "black", fg    = "white")
    self.label2.place(x=1080, y=78)
    
    self.label3 = Label(self.window, text = "Lesiòn tejido blando", bg= "black", fg = "white")
    self.label3.place(x=850, y=310)

    self.label4 = Label(self.window, text = "Tejido anexo", bg= "black",
    fg = "white")
    self.label4.place(x=1080, y=310)

if __name__ == "_main_": 
    root = Tk()
    inicio = Window(root) 
    root.mainloop()