

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

def csvEmpty():           
    f = open(path1+'/data_applegrading.csv', "w+")
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
        print('> loaded %s ' % (filename))
    
    
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
       
      
        with open(path1+'/data_orangesrading.csv','a') as f:
            savetxt(f,flat_features3,delimiter=',',fmt='%.18e')
            print('written done')
    feature_matrix = np.array(features_list)
    feature_matrix1 = feature_matrix[:,0,:]
    return feature_matrix1

def Create_Model(feature_matrix1):
  
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
    
    
    
    
    
    
    
    
    
    
path='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/dataset/oranges grading'
path3='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/test'
path1='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)'
path2='C:/Users/qadri/Desktop/major project/program/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/threshholdedreal1/'

labels = pd.read_csv(path1+"/labelsoranges.csv",index_col=0)
convert_Image(path)
csvEmpty()    
feature_matrix1=preProcessFeatures(path)
svm,accuracy=Create_Model(feature_matrix1)
print(testSVM(svm,path3))
    
 loaded_images_test = list()
    resized_imagestest= list()
    features_listtest=[]
    
    for filename in listdir(path3):    
        imagetest = Image.open(path3+'/'+filename)    
        loaded_images_test.append(imagetest)
        print('> loaded %s ' % (filename))  
        #dataarr = asarray(gs_image)
        #kholo = gs_image.show()
        resizedImagetest = imagetest.resize((100,100))
        resized_imagestest.append(resizedImagetest)
        reimagearrtest = asarray(resizedImagetest)
       # reimagearr1 = np.array(resizedImage)
        color_featurestest = reimagearrtest.flatten()
        color_features1test = np.float64(color_featurestest)
        gs_image_test = resizedImagetest.convert(mode='L')
        dataarrtest = asarray(gs_image_test)
        RET2,output2test=cv2.threshold(dataarrtest, th, max_val, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
        RET3,output3test=cv2.threshold(dataarrtest, th, max_val, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
        saveImgTrunctest = Image.fromarray(output2test)
        saveImgZInvtest = Image.fromarray(output3test)
       # saveImgTrunc.show()
        #saveImgZInv.show()
        #hog_features, hog_image = hog(dataarr,visualize=True,block_norm='L2-Hys',pixels_per_cell=(16, 16))
        #hog_features1, hog_image1 = hog(output2,visualize=True,block_norm='L2-Hys',pixels_per_cell=(16, 16))
        hog_features2test, hog_image2test = hog(output3test,visualize=True,block_norm='L2-Hys',pixels_per_cell=(16, 16))
        #hog_arr = Image.fromarray(hog_image)
        #hog_arr1 = Image.fromarray(hog_image1)
        
        flat_features1test=np.concatenate((color_features1test,hog_features2test), axis=None)
        flat_features2test=flat_features1test.reshape(-1, 1)
        flat_features3test=flat_features2test.transpose()
        #flat_features2=flat_features1.transpose((0, 0))
        #flat_features = np.hstack(color_features)
       
        features_listtest.append(flat_features3test)
        
    feature_matrixtest = np.array(features_listtest)
    feature_matrix2 = feature_matrixtest[:,0,:]    