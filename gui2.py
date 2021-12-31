
#from grading import *
from tkinter import *
import pickle  
import tkinter.filedialog as fd
from PIL import Image
import cv2
path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
path='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/gui'
path2='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/GUI/camera/'

from PIL import Image, ImageTk



class Window(Frame):

    
    def __init__(self, master=None):
        
       
        Frame.__init__(self, master)   
               
        self.master = master

        self.init_window()


    def init_window(self):

        self.master.title("Fruit Classification and Grading")

 
        self.pack(fill=BOTH, expand=1)
        SVMinit = Button(self, text="Initiate SVM",command=self.InitSVM,height = 3, width = 25)
    
        SVMinit.place(x=300,y=700)
        gradingbuttonB = Button(self, text="Banana Grading",command=self.filereadB,height = 3, width = 25)
        gradingbuttonB.place(x=500,y=700)
        gradingbuttonA = Button(self, text="Apple Grading",command=self.filereadA,height = 3, width = 25)
        gradingbuttonA.place(x=700,y=700)
        gradingbuttonO = Button(self, text="Orange Grading",command=self.filereadO,height = 3, width = 25)
        gradingbuttonO.place(x=900,y=700)
        classbutton = Button(self, text="Choose Image for Classification",command=self.filereadCL,height = 3, width = 25)
        classbutton.place(x=1100,y=700)
       
       
    def Camera(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Click Image")

        img_counter = 0
    
        while True:
            ret, frame = cam.read()
            cv2.imshow("Click Image", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
        
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "camera_{}.jpg".format(img_counter)
                
                cam.release()
                cv2.destroyAllWindows()
                #cameraCL(frame)
               # print("{} written!".format(img_name))
                img_counter += 1
                im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(im_rgb).save(path2+img_name)
                image=Image.fromarray(im_rgb)
                image1=image.resize((300,300))
                predicted=testSVMImageClass(loaded_svmcl,image)
         
                
                text= numtonameSingleClass(predicted) 
                output1 = Label(self, text="              ")
                output1.pack()
                output1.place(x=300,y=800)
                output = Label(self, text=text)
                output.pack()
                output.place(x=300,y=800)
                render = ImageTk.PhotoImage(image1)
                if(text=='APPLE'):
                    grapredicted=testSVMImageApple(loaded_svma,image)
             
                    
                    textgra= numtonameSingle(grapredicted) 
                    output2 = Label(self, text="              ")
                    output2.pack()
                    output2.place(x=300,y=820)
                    output = Label(self, text=textgra)
                    output.pack()
                    output.place(x=300,y=820)
                elif(text=='BANANA'):
                    grbpredicted=testSVMImageBanana(loaded_svmb,image)
             
                    
                    textgrb= numtonameSingle(grbpredicted) 
                    output2 = Label(self, text="              ")
                    output2.pack()
                    output2.place(x=300,y=820)
                    output = Label(self, text=textgrb)
                    output.pack()
                    output.place(x=300,y=820)  
                else:
                    gropredicted=testSVMImageOrange(loaded_svmo,image)
             
                    
                    textgro= numtonameSingle(gropredicted) 
                    output2 = Label(self, text="              ")
                    output2.pack()
                    output2.place(x=300,y=820)
                    output = Label(self, text=textgro)
                    output.pack()
                    output.place(x=300,y=820)     
        
                
                img = Label(self, image=render)
                img.image = render
                img.place(x=700, y=100)         
    
        cam.release()
        
        cv2.destroyAllWindows()
    
    def InitSVM(self):
        loaded_svmb=pickle.load(open(path1+'/bananasvmmodel.sav','rb'))
        loaded_svma=pickle.load(open(path1+'/applesvmmodel.sav','rb'))
        loaded_svmo=pickle.load(open(path1+'/orangesvmmodel.sav','rb'))
        loaded_svmcl=pickle.load(open(path1+'/classifier.sav','rb'))
        text = Label(self, text="INITIALIZATION SUCCESSFUL")
        text.pack()
        text.place(x=300,y=800)
        
    
    
    
    def showImg(self):
        load = Image.open(path+"/banana_fresh_0.jpg")
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)


    def filereadB(self):
        file=fd.askopenfile(mode="r")
       
        image = Image.open(file.name)
        image1=image.resize((300,300))
        predicted=testSVMImageBanana(loaded_svmb,image)
 
        
        text= numtonameSingle(predicted) 
        output1 = Label(self, text="              ")
        output1.pack()
        output1.place(x=500,y=800)
        output = Label(self, text=text)
        output.pack()
        output.place(x=500,y=800)
        render = ImageTk.PhotoImage(image1)

        
        img = Label(self, image=render)
        img.image = render
        img.place(x=700, y=100)  
        
    def filereadA(self):
        file=fd.askopenfile(mode="r")
       
        image = Image.open(file.name)
        image1=image.resize((300,300))
        predicted=testSVMImageApple(loaded_svma,image)
 
        
        text= numtonameSingle(predicted) 
        output1 = Label(self, text="              ")
        output1.pack()
        output1.place(x=700,y=800)
        output = Label(self, text=text)
        output.pack()
        output.place(x=700,y=800)
        render = ImageTk.PhotoImage(image1)

        
        img = Label(self, image=render)
        img.image = render
        img.place(x=700, y=100)     
        
        
    def filereadO(self):
        file=fd.askopenfile(mode="r")
       
        image = Image.open(file.name)
        image1=image.resize((300,300))
        predicted=testSVMImageOrange(loaded_svmo,image)
 
        
        text= numtonameSingle(predicted) 
        output1 = Label(self, text="              ")
        output1.pack()
        output1.place(x=900,y=800)
        output = Label(self, text=text)
        output.pack()
        output.place(x=900,y=800)
        render = ImageTk.PhotoImage(image1)

        
        img = Label(self, image=render)
        img.image = render
        img.place(x=700, y=100)   
        
    def filereadCL(self):
        file=fd.askopenfile(mode="r")
       
        image = Image.open(file.name)
        image1=image.resize((300,300))
        predicted=testSVMImageClass(loaded_svmcl,image)
 
        
        text= numtonameSingleClass(predicted) 
        output1 = Label(self, text="              ")
        output1.pack()
        output1.place(x=1100,y=800)
        output = Label(self, text=text)
        output.pack()
        output.place(x=1100,y=800)
        render = ImageTk.PhotoImage(image1)
        if(text=='APPLE'):
            grapredicted=testSVMImageApple(loaded_svma,image)
     
            
            textgra= numtonameSingle(grapredicted) 
            output2 = Label(self, text="              ")
            output2.pack()
            output2.place(x=1100,y=820)
            output = Label(self, text=textgra)
            output.pack()
            output.place(x=1100,y=820)
        elif(text=='BANANA'):
            grbpredicted=testSVMImageBanana(loaded_svmb,image)
     
            
            textgrb= numtonameSingle(grbpredicted) 
            output2 = Label(self, text="              ")
            output2.pack()
            output2.place(x=1100,y=820)
            output = Label(self, text=textgrb)
            output.pack()
            output.place(x=1100,y=820)  
        else:
            gropredicted=testSVMImageOrange(loaded_svmo,image)
     
            
            textgro= numtonameSingle(gropredicted) 
            output2 = Label(self, text="              ")
            output2.pack()
            output2.place(x=1100,y=820)
            output = Label(self, text=textgro)
            output.pack()
            output.place(x=1100,y=820)     

        
        img = Label(self, image=render)
        img.image = render
        img.place(x=700, y=100)   


       
       
        
       

root = Tk()
loaded_svmb=pickle.load(open(path1+'/bananasvmmodel.sav','rb'))
loaded_svma=pickle.load(open(path1+'/applesvmmodel.sav','rb'))
loaded_svmo=pickle.load(open(path1+'/orangesvmmodel.sav','rb'))
loaded_svmcl=pickle.load(open(path1+'/classifier.sav','rb'))
root.geometry("1600x900")


app = Window(root)



root.mainloop()