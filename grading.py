



from numpy import savetxt
from os import listdir
from PIL import Image
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle 
# load the image

def convert_Image(path):
    
    from os import listdir
    from PIL import Image
    for filename in listdir(path):
        image = Image.open(path+'/'+filename)
        if image.mode=='RGBA':
        
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask = image.split()[3])
            background.save(path+'/'+filename, "JPEG", quality=100)
            print(filename+' done RGBA->RGB')

        if image.mode=='P':
            image = image.convert('RGBA')
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask = image.split()[3])
            background.save(path+'/'+filename, "JPEG", quality=100)
            print(filename+' done P->RGB')

def csvEmpty(str):           
    f = open(path1+'/'+str, "w+")
    f.truncate()
    f.close()
    
def preProcessFeatures(path):  
    loaded_images = list()
    resized_images= list()
    
    th=0
    max_val=255
    features_list = []     
    for filename in listdir(path):    
        image = Image.open(path+'/'+filename)    
        loaded_images.append(image)
        #print('> loaded %s ' % (filename))
    
    
        resizedImage = image.resize((100,100))
        resized_images.append(resizedImage)
        reimagearr = asarray(resizedImage)

        color_features = reimagearr.flatten()
        color_features1 = np.float64(color_features)
        gs_image = resizedImage.convert(mode='L')
        dataarr = asarray(gs_image)
        RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        thresholded1=output2.flatten()
        RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
        hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
        hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
        flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
        flat_features2=flat_features1.reshape(-1, 1)
        flat_features3=flat_features2.transpose()
        features_list.append(flat_features3)
       
      
        with open(path1+'/data_testgrading.csv','a') as f:
            savetxt(f,flat_features3,delimiter=',',fmt='%.18e')
            print(filename+' done')
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    return feature_matrix1

def preProcessFeaturesImageB(image):  
   
    th=0
    max_val=255
    features_list = []     
    
    path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
        #print('> loaded %s ' % (filename))
    
    
    resizedImage = image.resize((100,100))
    
    reimagearr = asarray(resizedImage)

    color_features = reimagearr.flatten()
    color_features1 = np.float64(color_features)
    gs_image = resizedImage.convert(mode='L')
    dataarr = asarray(gs_image)
    RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    thresholded1=output2.flatten()
    RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
    hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
    hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
    flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
    flat_features2=flat_features1.reshape(-1, 1)
    flat_features3=flat_features2.transpose()
    features_list.append(flat_features3)
       
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    scaler_data_ = np.load(path1+'/'+"my_scaler_Banana.npy")
    mean1, vari1 = scaler_data_[0], scaler_data_[1]
    stds=np.sqrt(vari1)
    standard_matrix=(feature_matrix1-mean1)/stds

    return standard_matrix


def preProcessFeaturesImageClass(image):  
   
    th=0
    max_val=255
    features_list = []     
    
    path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
        #print('> loaded %s ' % (filename))
    
    #image=Image.open(path1+'/dataset/train/apple_fresh_0.jpg')
    resizedImage = image.resize((100,100))
    
    reimagearr = asarray(resizedImage)

    color_features = reimagearr.flatten()
    color_features1 = np.float64(color_features)
    gs_image = resizedImage.convert(mode='L')
    dataarr = asarray(gs_image)
    RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    thresholded1=output2.flatten()
    RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
    hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(8, 8))
    hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(8, 8))
                                        
    flat_features1=np.concatenate((color_features1,thresholded1,hog_features2),axis=None)
    flat_features2=flat_features1.reshape(-1, 1)
    flat_features3=flat_features2.transpose()
    features_list.append(flat_features3)
       
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]


    scaler_data_ = np.load(path1+'/'+"my_scaler_Classifier.npy")
    mean1, vari1 = scaler_data_[0], scaler_data_[1]
    stds=np.sqrt(vari1)
    standard_matrix=(feature_matrix1-mean1)/stds

    return standard_matrix
def preProcessFeaturesImageA(image):  
   
    th=0
    max_val=255
    features_list = []     
    
    path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
        #print('> loaded %s ' % (filename))
    
    
    resizedImage = image.resize((100,100))
    
    reimagearr = asarray(resizedImage)

    color_features = reimagearr.flatten()
    color_features1 = np.float64(color_features)
    gs_image = resizedImage.convert(mode='L')
    dataarr = asarray(gs_image)
    RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    thresholded1=output2.flatten()
    RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
    hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
    hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
    flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
    flat_features2=flat_features1.reshape(-1, 1)
    flat_features3=flat_features2.transpose()
    features_list.append(flat_features3)
       
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    scaler_data_ = np.load(path1+'/'+"my_scaler_Apple.npy")
    mean1, vari1 = scaler_data_[0], scaler_data_[1]
    stds=np.sqrt(vari1)
    standard_matrix=(feature_matrix1-mean1)/stds

    return standard_matrix
def preProcessFeaturesImageO(image):  
   
    th=0
    max_val=255
    features_list = []     
    
    path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
        #print('> loaded %s ' % (filename))
    
    
    resizedImage = image.resize((100,100))
    
    reimagearr = asarray(resizedImage)

    color_features = reimagearr.flatten()
    color_features1 = np.float64(color_features)
    gs_image = resizedImage.convert(mode='L')
    dataarr = asarray(gs_image)
    RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    thresholded1=output2.flatten()
    RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
    hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
    hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
    flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
    flat_features2=flat_features1.reshape(-1, 1)
    flat_features3=flat_features2.transpose()
    features_list.append(flat_features3)
       
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    scaler_data_ = np.load(path1+'/'+"my_scaler_Orange.npy")
    mean1, vari1 = scaler_data_[0], scaler_data_[1]
    stds=np.sqrt(vari1)
    standard_matrix=(feature_matrix1-mean1)/stds

    return standard_matrix
def preProcessFeaturesBanana(path):  
    loaded_images = list()
    resized_images= list()
    
    th=0
    max_val=255
    features_list = []     
    for filename in listdir(path):    
        image = Image.open(path+'/'+filename)    
        loaded_images.append(image)
        #print('> loaded %s ' % (filename))
    
    
        resizedImage = image.resize((100,100))
        resized_images.append(resizedImage)
        reimagearr = asarray(resizedImage)

        color_features = reimagearr.flatten()
        color_features1 = np.float64(color_features)
        gs_image = resizedImage.convert(mode='L')
        dataarr = asarray(gs_image)
        RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        thresholded1=output2.flatten()
        RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
        hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
        hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
        flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
        flat_features2=flat_features1.reshape(-1, 1)
        flat_features3=flat_features2.transpose()
        features_list.append(flat_features3)
       
      
        with open(path1+'/data_bananagrading.csv','a') as f:
            savetxt(f,flat_features3,delimiter=',',fmt='%.18e')
            print(filename+' done')
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    return feature_matrix1

def preProcessFeaturesOrange(path):
    loaded_images = list()
    resized_images= list()
    
    th=0
    max_val=255
    features_list = []     
    for filename in listdir(path):    
        image = Image.open(path+'/'+filename)    
        loaded_images.append(image)
        #print('> loaded %s ' % (filename))
    
    
        resizedImage = image.resize((100,100))
        resized_images.append(resizedImage)
        reimagearr = asarray(resizedImage)

        color_features = reimagearr.flatten()
        color_features1 = np.float64(color_features)
        gs_image = resizedImage.convert(mode='L')
        dataarr = asarray(gs_image)
        RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        thresholded1=output2.flatten()
        RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        thresholded2=output3.flatten()
   
        hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
        hog_features1, hog_image1 = hog(output3,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
        flat_features1=np.concatenate((color_features1,hog_features1,thresholded2),axis=None)
        flat_features2=flat_features1.reshape(-1, 1)
        flat_features3=flat_features2.transpose()
        features_list.append(flat_features3)
       
      
        with open(path1+'/data_orangesgrading.csv','a') as f:
            savetxt(f,flat_features3,delimiter=',',fmt='%.18e')
            print(filename+' done')
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    return feature_matrix1

def preProcessFeaturesApple(path):  
    loaded_images = list()
    resized_images= list()
    
    th=0
    max_val=255
    features_list = []     
    for filename in listdir(path):    
        image = Image.open(path+'/'+filename)    
        loaded_images.append(image)
        #print('> loaded %s ' % (filename))
    
    
        resizedImage = image.resize((100,100))
        resized_images.append(resizedImage)
        reimagearr = asarray(resizedImage)

        color_features = reimagearr.flatten()
        color_features1 = np.float64(color_features)
        gs_image = resizedImage.convert(mode='L')
        dataarr = asarray(gs_image)
        RET2,output2=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        thresholded1=output2.flatten()
        RET3,output3=cv2.threshold(dataarr, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
   
   
        hog_features2, hog_image2 = hog(reimagearr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
        hog_features1, hog_image1 = hog(gs_image,visualize=True,block_norm='L2-Hys',pixels_per_cell=(12, 12))
                                        
        flat_features1=np.concatenate((color_features1,hog_features2,thresholded1),axis=None)
        flat_features2=flat_features1.reshape(-1, 1)
        flat_features3=flat_features2.transpose()
        features_list.append(flat_features3)
       
      
        with open(path1+'/data_bananagrading.csv','a') as f:
            savetxt(f,flat_features3,delimiter=',',fmt='%.18e')
            print(filename+' done')
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    return feature_matrix1

def Create_Model(feature_matrix1,labels):
  
    ss = StandardScaler()
    # run this on our feature matrix
    fruits_stand= ss.fit_transform(feature_matrix1)
    
    pca = PCA(n_components=223)
    # use fit_transform to run PCA on our standardized matrix
    fruits_pca = ss.fit_transform(fruits_stand)
    X = pd.DataFrame(fruits_pca)
    #y = pd.Series(labels.genus.values)
    y = labels.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25122124,random_state=1234123)
    #3174324134
    pd.Series(y_train).value_counts()
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)


    y_pred = svm.predict(X_test)
    
    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)
    #print("Precision:",precision_score(y_test, y_pred,pos_label='positive',average='weighted'))
    #print("Recall:",recall_score(y_test, y_pred, average='macro'))
# look at the distrubution of labels in the train set
    probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
    y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
    roc_auc = auc(false_positive_rate, true_positive_rate)
#roc_auc1 = auc(y_test,y_pred)
    plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
    roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

    plt.legend(loc=0)
    plt.plot([0,1], [0,1], ls='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate');
    X_test1=asarray(X_test.index)
    X_train1=asarray(X_train.index)
    #x_test2=x_test1[:,0]
    plt.scatter(X_test1, y_test, color = 'red')
    plt.plot(X_test1, y_pred, color = 'green')
    plt.title('test set)')
    plt.ylabel('fruit')
    plt.xlabel('index')
    plt.show()
    
    
    return svm,accuracy


def testSVMImageBanana(svm,image):
    feature_matrix2=preProcessFeaturesImageB(image)
    
    X_TEST1 = pd.DataFrame(feature_matrix2) 
    y_pred1 = svm.predict(X_TEST1)  
    return y_pred1
def testSVMImageClass(svm,image):
    feature_matrix2=preProcessFeaturesImageClass(image)
    
    X_TEST1 = pd.DataFrame(feature_matrix2) 
    y_pred1 = svm.predict(X_TEST1)  
    return y_pred1
def testSVMImageApple(svm,image):
    feature_matrix2=preProcessFeaturesImageA(image)
    
    X_TEST1 = pd.DataFrame(feature_matrix2) 
    y_pred1 = svm.predict(X_TEST1)  
    return y_pred1
def testSVMImageOrange(svm,image):
    feature_matrix2=preProcessFeaturesImageO(image)
    
    X_TEST1 = pd.DataFrame(feature_matrix2) 
    y_pred1 = svm.predict(X_TEST1)  
    return y_pred1
def testSVM(svm,path3):
    feature_matrix2=preProcessFeatures(path3)
    ss1 = StandardScaler()
    # run this on our feature matrix
    fruits_stand1= ss1.fit_transform(feature_matrix2) 
    pca1 = PCA(n_components=8)
    # use fit_transform to run PCA on our standardized matrix
    fruits_pca1 = ss1.fit_transform(fruits_stand1)
    X_TEST1 = pd.DataFrame(fruits_pca1) 
    y_pred1 = svm.predict(X_TEST1)  
    return y_pred1
    
    
    
def numtoname(y_pred):
    y_prednames=[]
    y=0
    for i in y_pred:
        if (i==1):
            y_prednames.append('FRESH')
            y=y+1
        else:
            y_prednames.append('ROTTEN')
            y=y+1
        
    y_prednames1=asarray(y_prednames)
    return y_prednames1

def numtonameSingle(y_pred):
    i=y_pred
    if (i==1):
        text='FRESH'
    else:
        text='ROTTEN'
        
    return text
def numtonameSingleClass(y_pred):
    i=y_pred
    if (i==2):
        text='APPLE'
    elif(i==1):
        text='BANANA'
    else:
        text='ORANGE'
        
    return text

    
    
   
  
if __name__ == "__main__": 
    
    convert_Image()
    csvEmpty()
    preProcessFeatures()
    preProcessFeaturesImageO()  
    preProcessFeaturesImageA()  
    preProcessFeaturesImageB()  
    preProcessFeaturesImageClass()
    preProcessFeaturesBanana()
    preProcessFeaturesOrange()
    preProcessFeaturesApple()
    Create_Model()
    testSVM()
    testSVMImageClass()
    testSVMImageBanana()
    testSVMImageApple()
    testSVMImageOrange()
    numtoname()
    numtonameSingle()
    numtonameSingleClass()
    patha='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/dataset/apple grading'
    path3='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/test'
    path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
    pathb='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/dataset/banana grading'
    patho='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/dataset/oranges grading'
    labelsa = pd.read_csv(path1+"/labelsapple.csv",index_col=0)
    labelsb = pd.read_csv(path1+"/labelsbanana.csv",index_col=0)
    labelso = pd.read_csv(path1+"/labelsoranges.csv",index_col=0)
    convert_Image(patha)
    convert_Image(pathb)
    convert_Image(patho)
    csvEmpty('data_applegrading.csv')    
    csvEmpty('data_bananagrading.csv')  
    csvEmpty('data_orangesgrading.csv')  
    feature_matrixapple=preProcessFeaturesApple(patha)
    feature_matrixbanana=preProcessFeaturesBanana(pathb)
    feature_matrixorange=preProcessFeaturesOrange(patho)
    svma,accuracya=Create_Model(feature_matrixapple,labelsa)
    svmb,accuracyb=Create_Model(feature_matrixbanana,labelsb)
    svmo,accuracyo=Create_Model(feature_matrixorange,labelso)
    pickle.dump(svmb,open(path1+'/bananasvmmodel.sav','wb'))
    pickle.dump(svma,open(path1+'/applesvmmodel.sav','wb'))
    pickle.dump(svmo,open(path1+'/orangesvmmodel.sav','wb'))
    loaded_svmb=pickle.load(open(path1+'/bananasvmmodel.sav','rb'))
    loaded_svma=pickle.load(open(path1+'/applesvmmodel.sav','rb'))
    loaded_svmo=pickle.load(open(path1+'/orangesvmmodel.sav','rb'))
    #print(testSVM(svma,path3))
    #print(testSVM(svmb,path3))
    y_pred=testSVM(loaded_svmb,path3)
    
    
     

