import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics as stat
from tkinter.filedialog import asksaveasfilename
import joblib

def MLRNormEq(savemodel,data,yname,testsize):
    
    try:
        frame = pd.read_excel(data)
        frame = frame.replace(r'^\s*$', np.nan, regex=True)
    except Exception:
        frame = pd.read_csv(data, sep=",")
        frame = frame.replace(r'^\s*$', np.nan, regex=True)
        
 
    frame = frame.dropna()
    yframe = frame[yname]
    Xframe = frame.drop([yname], axis = 1)
    Xframe['Intercept'] = 1
    X = Xframe.values
    y = yframe.values
    
    actual = "Actual"+" "+yname
    predicted = "Predicted"+" "+yname
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsize)
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.dot(np.transpose(X_train),y_train))
    thetas  = np.transpose(theta.reshape(-1,1))
    ypredtest = np.dot(X_test,theta)
    ypredtrain = np.dot(X_train,theta)
    model = pd.DataFrame(data = thetas, columns = Xframe.columns)
    
    if savemodel == 1:
        modelname = asksaveasfilename(filetypes=(("Pickle files", "*.pkl"),("All files", "*.*") )) 
        joblib.dump(model,modelname+'.pkl')
    else:
        pass
    
    avedifftest = stat.mean(abs(y_test-ypredtest))
    avedifftrain = stat.mean(abs(y_train-ypredtrain))
    
    traincorr = np.corrcoef(ypredtrain,y_train)
    traincorr2 = traincorr[0,1]
    train_score = traincorr2**2
    
    testcorr = np.corrcoef(ypredtest,y_test)
    testcorr2 = testcorr[0,1]
    test_score = testcorr2**2
    
    
    avetrainerr = stat.mean(abs(y_train-ypredtrain)/y_train)
    avetesterr = stat.mean(abs(y_test-ypredtest)/y_test)
    
    difftest = abs(y_test-ypredtest)
    difftrain = abs(y_train-ypredtrain)
    
    testerr = abs(y_test-ypredtest)/y_test
    trainerr = abs(y_train-ypredtrain)/y_train
    
    train_data = pd.DataFrame(data = X_train, columns = Xframe.columns)

    train_data[actual] = y_train
    train_data[predicted] = ypredtrain
    train_data['Error'] = trainerr
    train_data['Abs Diff'] = difftrain
    test_data = pd.DataFrame(data = X_test, columns = Xframe.columns)
    test_data[actual] = y_test
    test_data[predicted] = ypredtest
    test_data['Error'] = testerr
    test_data['Abs Diff'] = difftest
    
    return (train_score,test_score,train_data,test_data,avetrainerr,avetesterr,avedifftrain,avedifftest)

