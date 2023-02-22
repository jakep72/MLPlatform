import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import statistics as stat
from tkinter.filedialog import asksaveasfilename
import joblib


def MLPRegress(savemodel, data, yname, testsize, numlayers, layersize,activ, tols,regs,iters):
       
    try:
        frame = pd.read_excel(data)
        frame = frame.replace(r'^\s*$', np.nan, regex=True)
    except Exception:
        frame = pd.read_csv(data, sep=",")
        frame = frame.replace(r'^\s*$', np.nan, regex=True)
        
    architect = (layersize,)*numlayers
    frame = frame.dropna()
    yframe = frame[yname]
    Xframe = frame.drop([yname], axis = 1)
    X = Xframe.values
    y = yframe.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsize)
    
    mlp = make_pipeline(StandardScaler(), MLPRegressor(activation = activ,hidden_layer_sizes=architect,max_iter=iters,tol = tols, alpha = regs))
    mlp.fit(X_train,y_train)

    
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test,y_test)
    ypredtest = mlp.predict(X_test)
    ypredtrain = mlp.predict(X_train)
    
    actual = "Actual"+" "+yname
    predicted = "Predicted"+" "+yname
    
    if savemodel == 1:
        modelname = asksaveasfilename(filetypes=(("Pickle files", "*.pkl"),
                                                         ("All files", "*.*") )) 
    
        joblib.dump(mlp,modelname+'.pkl')
    else:
        pass
    
    avetesterr = stat.mean(abs(y_test-ypredtest)/y_test)
    avetrainerr = stat.mean(abs(y_train-ypredtrain)/y_train)
    
    avedifftest = stat.mean(abs(y_test-ypredtest))
    avedifftrain = stat.mean(abs(y_train-ypredtrain))
    
    testerr = abs(y_test-ypredtest)/y_test
    trainerr = abs(y_train-ypredtrain)/y_train
    
    difftest = abs(y_test-ypredtest)
    difftrain = abs(y_train-ypredtrain)
    
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

