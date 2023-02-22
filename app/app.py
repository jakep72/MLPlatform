import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter.font as font
from tkinter import messagebox
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import mplcursors
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.ticker import MaxNLocator
from mlplatform.core.mlpRegress import MLPRegress
from mlplatform.core.mlrNormEq import MLRNormEq
from mlplatform.core.data_manipulation import create_report, create_filtered, transfer
import mlplatform.pandastable_mods as pandastable
from mlplatform.pandastable_mods import Table, TableModel

# Global style and format settings
LARGE_FONT= ("Verdana", 12)
style.use("seaborn-v0_8")

f1 = Figure(figsize = (5,5), dpi = 100)
a = f1.add_subplot(111)

f2 = Figure(figsize = (5,5), dpi = 100)
b = f2.add_subplot(111)

f3 = Figure(figsize=(7, 6), dpi=80)
ax2 = f3.add_subplot()
    
class MLPlatform(tk.Tk):
    """
    MLPlatform is a user interface meant to simplify exploring and cleaning datasets and provides useful tools that can be used to model the data.
    The tkinter-based interface currently implements linear regression modelling via the normal equation and supports scikit-learn multi layer
    perceptron models.  Datasets can be visualized in the data exploration tab and cleaned datasets can be saved and used to build models.  Summary
    training results and the model weights can be saved locally for analysis and for further use.
    """
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        
        tk.Tk.wm_title(self, "ML Platform")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, Regression, Classification, MLR, MLPRegressor):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):
    """
    Main application landing page.  Classifiers are currently under construction.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent,bg="#26242f")
        myFont = font.Font(size = 15, weight = 'bold')

        regr = tk.Button(self, text="Regression",
                            command=lambda: [controller.show_frame(Regression)], height = 3, width = 30, bd = 5, fg = 'black', bg ='gray')
        regr['font'] = myFont
        regr.pack(padx = 25, pady = 15)
        
        classi = tk.Button(self, text="Classification",
                            command=lambda: [controller.show_frame(Classification)], height = 3, width = 30, bd = 5, fg = 'black', bg ='gray')
        classi['font'] = myFont                     
        classi.pack(padx = 25, pady = 15)
        
class Regression(tk.Frame):
    """
    Regression landing page.  Multiple linear regression and multi layer perceptrons are currently supported.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#26242f")
        myFont = font.Font(size = 15, weight = 'bold')
        home = tk.Button(self, text="Back to Home",
                            command=lambda: [controller.show_frame(StartPage)],bd = 2, fg = 'black', bg ='gray')
        home.pack(pady = 5)
        
        MLRbutton = tk.Button(self, text="Multiple Linear Regression",
                            command=lambda: [controller.show_frame(MLR)], height = 5, width = 30, bd = 5, fg = 'black', bg ='gray')
        MLRbutton['font'] = myFont
        MLRbutton.pack(padx = 25, pady = 25)

        MLPbutton = tk.Button(self, text="Multi Layer Perceptron",
                            command=lambda: [controller.show_frame(MLPRegressor)], height = 5, width = 30, bd = 5, fg = 'black', bg ='gray')
        MLPbutton['font'] = myFont
        MLPbutton.pack(padx = 25, pady = 25)        
        

class Classification(tk.Frame):
    """
    Classification landing page
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#26242f")

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: [controller.show_frame(StartPage)], bd = 2, fg = 'black', bg ='gray')
        button1.pack(pady = 5)
        
        label = tk.Label(self, text = 'Features Coming Soon...', pady = 10, font = ('Helvetica', 18, 'bold'), bg="#26242f",fg='white')
        label.pack()
        

        
class MLR(tk.Frame):
    """
    Multiple linear regression landing page.

    Use multiple linear regression to model linear interactions between input and output data.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)       
        myFont = font.Font(size = 15, weight = 'bold')
        sFont = font.Font(size = 10, weight = 'bold')
        
        def reset():
            # Clear the page of all buttons, labels, graphics, etc
            for widget in self.winfo_children():
                widget.destroy()
                b.clear()
                                
        def create(self): 
            # Create the multiple linear regression page       
            back = tk.Button(self, text="Back to Regression Page",
                                command=lambda: [controller.show_frame(Regression),reset(),create(self)], bd = 2, fg = 'black', bg ='gray')
            back.pack(pady = 5)
                      
            frame1 = tk.Frame(self)
            frame1.pack(fill=X)
            
            MLRfile=tk.Entry(frame1)
            MLRfile.config(width = 40)
    
            def get_cols():
                # Return the names of the data columns
                data = MLRfile.get()
                try:
                    MLRframe = pd.read_excel(data)
                except Exception:
                    MLRframe = pd.read_csv(data, sep=",")
                cols = list(MLRframe.columns)
                return (cols)
            
            def SelectFile():
                # Ask user for input file
                MLRfile.delete(0,'end')
                filename = filedialog.askopenfilename(filetypes = [("all files", "*.*"),("CSV files","*.CSV"),("xls files","*xls"),("xlsx files","*.xlsx")])
                MLRfile.insert(tk.END,filename)
                pred1['menu'].delete(0,'end')
                global MLRpredvar

                predchoices = get_cols()

                
                for choice in predchoices:
                    pred1['menu'].add_command(label = choice,command = tk._setit(MLRpredvar,choice))
                    
                MLRpredvar.set("Choose Variable")
                b1.configure(bg = 'green')

                            
            b1=tk.Button(frame1,text="Select a Data File",command=SelectFile,bd = 5, bg = 'red', fg ='#ffffff')
            b1.pack(side = LEFT,pady = 5, padx = 5)
            MLRfile.pack(side = LEFT, pady =5, padx = 5)
            
            
            def create_top():
                # Create the data exploration page
                pandastable.dialogs.QueryDialog.hello.clear()
                data = MLRfile.get()
                global MLRframe
                try:
                    MLRframe = pd.read_excel(data)
                except Exception:
                    MLRframe = pd.read_csv(data, sep=",")
                collist = list(MLRframe.columns)
                
                ecols = pd.DataFrame(data = MLRframe.columns, columns = ['Columns'])
                is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
                types = pd.DataFrame(data = is_number(MLRframe.dtypes), columns = ['Numeric'])
                
                efilt = ecols.merge(types, left_index = True, right_index = True)
                efilt = efilt[efilt['Numeric']==False]
                vals = efilt['Columns'].tolist()
                vals = 'The Data Column(s) '+' and '.join(vals) + ' contain(s) non-numeric data. This data must be altered before the regression is fit.'

                try:                   
                    if not efilt.empty:
                        raise Exception('value')
                    elif efilt.empty:
                        pass
                except Exception:
                    messagebox.showerror('Data Error',vals)
                    
                top2 = tk.Toplevel()
                top2.geometry("850x850")
                top2.wm_title("Data Exploration")
                
                frame04 = tk.Frame(top2)
                frame04.pack(side = TOP, pady = 2)
                frame03 = tk.Frame(top2)
                frame03.pack(side = TOP, pady = 2)
                frame02 = tk.Frame(top2)
                frame02.pack(side = TOP, pady = 2)
                frame00 = tk.Frame(top2)
                frame00.pack(side = TOP,fill=X, pady = 5)
                frame0175 = tk.Frame(top2)
                frame0175.pack(side = TOP, pady = 2)
                frame015 = tk.Frame(top2)
                frame015.pack(side = TOP, pady = 2)
                frame01 = tk.Frame(top2)
                frame01.pack(side = TOP, pady = 2)
                frame0 = tk.Frame(top2)
                frame0.pack(side = TOP, pady = 5)

                lbl3 = tk.Label(frame04, text = "Choose a Data View",font = ('Helvetica', 14, 'bold'))
                fo = font.Font(lbl3, label.cget("font"))
                fo.configure(underline=True)
                lbl3.configure(font=fo)
                lbl3.pack()
                lbl4 = tk.Label(frame0175, text = "Data Filtering",font = ('Helvetica', 14, 'bold'))
                fo1 = font.Font(lbl4, label.cget("font"))
                fo1.configure(underline=True)
                lbl4.configure(font=fo)
                lbl4.pack()
                
                v = tk.IntVar()

    
                rb0 = tk.Radiobutton(frame03, text = "Data Summary", variable = v, value = 0).pack(side =LEFT)
                rb1 = tk.Radiobutton(frame03, text = "Correlation Plot", variable = v, value = 1).pack(side=LEFT)
                rb2 = tk.Radiobutton(frame03, text = "Scatter Plots", variable = v, value = 2).pack(side =LEFT)
                                  
                MLRdf = TableModel(MLRframe)
                MLRpt = Table(frame015,MLRdf)
                global MLRfb
                MLRfb = pandastable.dialogs.QueryDialog(MLRpt)

                
                def convert():
                    # Convert the pandas dataframe to a pandastable Table model
                    global MLRframe
                    global MLRrd

                    MLRrd = MLRfb.getdf()
                    try:
                        MLRrd = MLRrd[0]
                    except Exception:
                        MLRrd = MLRframe
                    return (MLRrd)
                    
                button1 = tk.Button(frame02, text = "Show / Update Data View", command = lambda: [clearframe(frame01,frame0),setDataView(v.get(),convert(),frame01,frame0,top2)],height = 1, width = 20 , bd = 5, fg = 'black', bg ='gray')
                button1['font'] = sFont
                button1.pack()
                MLRfb.pack()
                global MLRrd
                snf = tk.Button(frame00, text = "Save Data to New File", command = lambda: [create_filtered(convert())],height = 1, width = 20 , bd = 5, fg = 'black', bg ='gray')
                snf['font'] = sFont
                snf.pack(side = LEFT, padx = 5, pady = 5)
                
                def save_update():
                    # Update the data and save as new file
                    path = create_filtered(convert())+'.xlsx'
                    MLRfile.delete(0,'end')
                    MLRfile.insert(tk.END,path)
                    collist2 = list(convert().columns)
                    pred1['menu'].delete(0,'end')
                    for choice in collist2:
                        pred1['menu'].add_command(label = choice,command = tk._setit(MLRpredvar,choice))
                    top2.destroy()
                    
                snuf = tk.Button(frame00, text = "Save Data & Update Inputs", command = lambda: [save_update()],height = 1, width = 25, bd = 5, fg = 'black', bg ='gray')
                snuf['font'] = sFont
                snuf.pack(side = RIGHT, padx = 5, pady = 5)
                
            def setDataView(view, data,tlevel,slevel,vtlevel):
                # Configure settings for the selected data view option on the data exploration page
                collist = list(data.columns)
                if view == 0:
                    
                    summ = data.describe()
                    summ = summ.reindex(index=summ.index[::-1])
    
                    cols = list(summ.columns)
                    global MLRtree
                    global MLRsb
                    MLRtree = ttk.Treeview(tlevel, selectmode = 'extended')
                    MLRtree.pack()
                    MLRsb = ttk.Scrollbar(tlevel, orient = "horizontal",command = MLRtree.xview)
                    MLRsb.pack(fill = X)
                    MLRtree.configure(xscrollcommand = MLRsb.set)
                    MLRtree["columns"] = cols
                    for i in cols:
                        MLRtree.column(i, anchor="w")
                        MLRtree.heading(i, text=i, anchor='w')
    
                    for index, row in summ.iterrows():
                        MLRtree.insert("",0,text=index,values=list(row))

                        
                elif view == 1:
                    ax2.clear()
                    corr = data.corr()
                    sns.heatmap(corr, annot = True, ax = ax2, cbar = False,cmap=sns.diverging_palette(20, 220, n=200))
                    canvas = FigureCanvasTkAgg(f3,tlevel)
                    canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True, padx = 10, pady = 10)
                
                elif view == 2:
                    
                    lbl = tk.Label(tlevel, text = "X-Axis:",font = ('Helvetica', 14, 'bold'))
                    lbl.pack(side = LEFT, padx = 5)
                    xvar = StringVar()
                    xax = tk.OptionMenu(tlevel,xvar,value =None)
                    xvar.set(collist[0])
                    xax.pack(side= LEFT)
                    
    
                    lbl2 = tk.Label(tlevel, text = "Y-Axis:",font = ('Helvetica', 14, 'bold'))
                    lbl2.pack(side = LEFT, padx = 5)
                    yvar = StringVar()
                    yax = tk.OptionMenu(tlevel,yvar,value =None)
                    yvar.set(collist[0])
                    yax.pack(side = LEFT)
                    def update(level):
                        
                        global MLRframe
                        global MLRrd
                        global MLRfb
                        MLRrd = MLRfb.getdf()
                        try:
                            data = MLRrd[0]
                            collist3 = data.columns
                            xax['menu'].delete(0,'end')
                            yax['menu'].delete(0,'end')
                            for choice in collist3:
                                xax['menu'].add_command(label = choice,command = tk._setit(xvar,choice))
                                yax['menu'].add_command(label = choice,command = tk._setit(yvar,choice))
                            
                        except Exception:
                            data = MLRframe
                            
                        ax2.clear()
                        ax2.scatter(data[xvar.get()],data[yvar.get()])
                        ax2.set_xlabel(xvar.get())
                        ax2.set_ylabel(yvar.get())
                        mplcursors.cursor(ax2, hover=True)
                        canvas = FigureCanvasTkAgg(f3,level)
                        f3.canvas.draw()
                        canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True, padx = 10, pady = 10)

                    upd = tk.Button(tlevel,text = "Update Graph", command = lambda: [cleargraph(slevel),update(slevel)],height = 1, width = 20, bd = 5, fg = 'black', bg ='gray')
                    upd['font'] = sFont
                    upd.pack(side = TOP, padx = 30)
                    
                    for choice in collist:
                        xax['menu'].add_command(label = choice,command = tk._setit(xvar,choice))
                        yax['menu'].add_command(label = choice,command = tk._setit(yvar,choice))
                    
            def clearframe(tlevel,slevel):
                # Clear the data exploration page
                for widget in tlevel.winfo_children():
                    widget.destroy()
                cleargraph(slevel)

            def cleargraph(slevel):
                # Clear the data view graphic
                for widget in slevel.winfo_children():
                    widget.destroy()
            
            frame25 = tk.Frame(self)
            frame25.pack(fill=X)
                
            vis = tk.Button(frame1, text = "Data Exploration", command = create_top, bd = 2, fg = 'black', bg ='gray')
            vis.pack_forget()
            
                
            label1 = tk.Label(frame25, text = "Variable to Predict:",font = ('Helvetica', 14, 'bold'))
            fo = font.Font(label1, label1.cget("font"))
            label1.pack(side = LEFT, pady = 5, padx = 5)
            label1.configure(font=fo)
            
            global MLRpredvar
            MLRpredvar = StringVar()
            pred1 = tk.OptionMenu(frame25,MLRpredvar,value = None)
            MLRpredvar.set("Select a Data File First")
            pred1.pack(side = LEFT)
            
            def visual(*args):
                vis.pack(side = LEFT, padx = 5, pady=5)
                
            MLRpredvar.trace('w',visual)

            
            frame3 = tk.Frame(self)
            frame3.pack(fill = X)
            
            label2 = tk.Label(frame3, text = "Fraction of Data to Test:")
            label2.pack(side = LEFT, padx = 5, pady = 5)
            label2.configure(font=fo)
            
            MLRfrac = tk.Entry(frame3,justify = 'center')
            MLRfrac.config(width = 19)
            MLRfrac.pack(side = LEFT,padx = 5,pady = 5)
            
            
            frame6 = tk.Frame(self)
            frame6.pack(fill=X)
            
            label = tk.Label(frame6, text = "Save Model for Later Use?")
            label.pack(side = LEFT,pady = 5, padx = 5)
            
            label.configure(font=fo)
    
            smr = tk.IntVar()
            smr.set(0)
    
            rb1 = tk.Radiobutton(frame6, text = "Yes", variable = smr, value = 1).pack(side =LEFT)
            rb2 = tk.Radiobutton(frame6, text = "No", variable = smr, value = 0).pack(side=LEFT) 
            
            
            frame7 = tk.Frame(self)
            frame7.pack(fill=X)
            
            def MLRFunc():
                # Use the normal equation to train a multiple linear regression model
                progvar.set('')
                data = str(MLRfile.get())               
                if data == '':
                    messagebox.showerror('File Not Found Error','Please select an input file')
                    return
                yname = str(MLRpredvar.get())
                if yname == 'Choose Variable':
                    messagebox.showerror('Variable to Predict Error','Please select the variable you would like to train the regression to predict')
                    return
                try:
                    testsize = float(MLRfrac.get())
                    if testsize >= 1 or testsize <= 0:
                        raise Exception('value')
                except Exception:       
                    messagebox.showerror('Test Size Error','Fraction of Data to Test must be a decimal between 0 and 1')
                    return
                    
                try:
                    edata = pd.read_excel(data)
                except Exception:
                    edata = pd.read_csv(data, sep=",")
                
                ecols = pd.DataFrame(data = edata.columns, columns = ['Columns'])
                is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
                types = pd.DataFrame(data = is_number(edata.dtypes), columns = ['Numeric'])
                
                efilt = ecols.merge(types, left_index = True, right_index = True)
                efilt = efilt[efilt['Numeric']==False]
                vals = efilt['Columns'].tolist()
                vals = 'The Data Column(s) '+' and '.join(vals) + ' contain(s) non-numeric data. This data can not be used in regression analyses.'

                try:                   
                    if not efilt.empty:
                        raise Exception('value')
                    elif efilt.empty:
                        pass
                except Exception:
                    messagebox.showerror('Data Error',vals)
                    return
                
                global actual
                global predicted
                
                actual = "Actual"+" "+yname
                predicted = "Predicted"+" "+yname
        
                global train_score
                global test_score
                global train_data
                global test_data
                global avetrainerr
                global avetesterr
                global avedifftrain
                global avedifftest
                
                             
                progvar.set('Training in Progress...')
                progress.configure(foreground = 'red')
                self.update_idletasks()
    
                train_score,test_score,train_data,test_data,avetrainerr,avetesterr,avedifftrain,avedifftest = MLRNormEq(smr.get(),data,yname,testsize)
         
  
                
                trainerr.config(text = "Average Train Error = "+str(round(avetrainerr,3)*100)+"%")
                testerr.config(text = "Average Test Error = "+str(round(avetesterr,3)*100)+"%")
                progvar.set('Training Complete!')
                progress.config(foreground = 'green')
                a.clear()
                
                a.set_xlabel(predicted)
                a.set_ylabel(actual)
                a.set_title("Test Dataset Performance")
                a.scatter(test_data[predicted],test_data[actual])
                mplcursors.cursor(a, hover=True)
    
        
                locator = MaxNLocator(prune = 'both', nbins = 20)
                a.xaxis.set_major_locator(locator)
                a.tick_params(axis = 'x', labelsize = 'small', rotation =  60)
    
                f1.canvas.draw() 
                
                    
            
            train=tk.Button(frame7,text="Train / Test",command=lambda: [MLRFunc()],height = 2, width = 20, bd = 5, fg = 'black', bg ='gray')
            train['font'] = myFont
            train.pack(side=LEFT,padx = 5,pady = 5)
                

                
                
            export=tk.Button(frame7,text="Export Report to Excel",command=lambda: [reports()],height = 2, width = 20, bd = 5, fg = 'black', bg ='gray')
            export['font']=myFont
            export.pack(side = RIGHT,padx = 5,pady = 5)
            def reports():
                try:    
                    create_report(train_data,test_data,train_score,test_score,avetrainerr,avetesterr,avedifftrain,avedifftest)
                except Exception:
                    messagebox.showerror('Error','No Report to Export, be sure to train the neural network first')
            frame8 = tk.Frame(self)
            frame8.pack(fill=X)
            
            trainerr = tk.Label(frame8,text = "Average Train Error  = ")
            trainerr.pack(side = LEFT,padx = 100,pady = 5)
            
            testerr = tk.Label(frame8,text = "Average Test Error  = ")
            testerr.pack(side=RIGHT,padx = 100,pady = 5)
            
            frame9 = tk.Frame(self)
            frame9.pack(fill=X)
            
            progvar = StringVar()
            progress = tk.Label(frame9,textvariable = progvar,font = ('Helvetica', 14, 'bold'))
            progress.pack()
                
            canvas = FigureCanvasTkAgg(f1,self)
            canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True)
        frame = create(self) 
        
        
class MLPRegressor(tk.Frame):
    """
    Multi layer perceptron landing page.

    Train a scikit-learn neural network to model input and output data.
    """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)       
        myFont = font.Font(size = 15, weight = 'bold')
        sFont = font.Font(size = 10, weight = 'bold')
        
        def reset():
            # Clear the page of all buttons, labels, graphics, etc
            for widget in self.winfo_children():
                widget.destroy()
                b.clear()
                                
        def create(self): 
            # Create the multi layer perceptron page       
            back = tk.Button(self, text="Back to Regression Page",
                                command=lambda: [controller.show_frame(Regression),reset(),create(self)], bd = 2, fg = 'black', bg ='gray')
            back.pack(pady = 5)
                      
            frame1 = tk.Frame(self)
            frame1.pack(fill=X)
            
            file=tk.Entry(frame1)
            file.config(width = 40)
    
            def get_cols():
                # Return the names of the data columns
                data = file.get()
                try:
                    frame = pd.read_excel(data)
                except Exception:
                    frame = pd.read_csv(data, sep=",")
                cols = list(frame.columns)
                return (cols)
            
            def SelectFile():
                # Ask user for input file
                file.delete(0,'end')
                filename = filedialog.askopenfilename(filetypes = [("all files", "*.*"),("CSV files","*.CSV"),("xls files","*xls"),("xlsx files","*.xlsx")])
                file.insert(tk.END,filename)
                pred1['menu'].delete(0,'end')
                global predvar

                predchoices = get_cols()

                
                for choice in predchoices:
                    pred1['menu'].add_command(label = choice,command = tk._setit(predvar,choice))
                    
                predvar.set("Choose Variable")
                b1.configure(bg = 'green')

                            
            b1=tk.Button(frame1,text="Select a Data File",command=SelectFile,bd = 5, bg = 'red', fg ='#ffffff')
            b1.pack(side = LEFT,pady = 5, padx = 5)
            file.pack(side = LEFT, pady =5, padx = 5)
            
            
            def create_top():
                # Create the data exploration page
                pandastable.dialogs.QueryDialog.hello.clear()
                data = file.get()
                global frame
                try:
                    frame = pd.read_excel(data)
                    frame = frame.replace(r'^\s*$', np.nan, regex=True)
                except Exception:
                    frame = pd.read_csv(data, sep=",")
                    frame = frame.replace(r'^\s*$', np.nan, regex=True)
                collist = list(frame.columns)
                
                ecols = pd.DataFrame(data = frame.columns, columns = ['Columns'])
                is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
                types = pd.DataFrame(data = is_number(frame.dtypes), columns = ['Numeric'])
                
                efilt = ecols.merge(types, left_index = True, right_index = True)
                efilt = efilt[efilt['Numeric']==False]
                vals = efilt['Columns'].tolist()
                vals = 'The Data Column(s) '+' and '.join(vals) + ' contain(s) non-numeric data. This data must be altered before the neural network can be trained.'

                try:                   
                    if not efilt.empty:
                        raise Exception('value')
                    elif efilt.empty:
                        pass
                except Exception:
                    messagebox.showerror('Data Error',vals)
                    
                
                top2 = tk.Toplevel()
                top2.geometry("850x850")
                top2.wm_title("Data Exploration")
                
                frame04 = tk.Frame(top2)
                frame04.pack(side = TOP, pady = 2)
                frame03 = tk.Frame(top2)
                frame03.pack(side = TOP, pady = 2)
                frame02 = tk.Frame(top2)
                frame02.pack(side = TOP, pady = 2)
                frame00 = tk.Frame(top2)
                frame00.pack(side = TOP,fill=X, pady = 5)
                frame0175 = tk.Frame(top2)
                frame0175.pack(side = TOP, pady = 2)
                frame015 = tk.Frame(top2)
                frame015.pack(side = TOP, pady = 2)
                frame01 = tk.Frame(top2)
                frame01.pack(side = TOP, pady = 2)
                frame0 = tk.Frame(top2)
                frame0.pack(side = TOP, fill = X, expand = True,pady = 5)

                lbl3 = tk.Label(frame04, text = "Choose a Data View",font = ('Helvetica', 14, 'bold'))
                fo = font.Font(lbl3, label.cget("font"))
                fo.configure(underline=True)
                lbl3.configure(font=fo)
                lbl3.pack()
                lbl4 = tk.Label(frame0175, text = "Data Filtering",font = ('Helvetica', 14, 'bold'))
                fo1 = font.Font(lbl4, label.cget("font"))
                fo1.configure(underline=True)
                lbl4.configure(font=fo)
                lbl4.pack()
                
                v = tk.IntVar()

    
                rb0 = tk.Radiobutton(frame03, text = "Data Summary", variable = v, value = 0).pack(side =LEFT)
                rb1 = tk.Radiobutton(frame03, text = "Correlation Plot", variable = v, value = 1).pack(side=LEFT)
                rb2 = tk.Radiobutton(frame03, text = "Scatter Plots", variable = v, value = 2).pack(side =LEFT)
                                  
                df = TableModel(frame)
                pt = Table(frame015,df)
                global fb
                fb = pandastable.dialogs.QueryDialog(pt)

                
                def convert():
                    # Convert the pandas dataframe to a pandastable Table model
                    global frame
                    global rd

                    rd = fb.getdf()
                    try:
                        rd = rd[0]
                    except Exception:
                        rd = frame
                    return (rd)
                    
                button1 = tk.Button(frame02, text = "Show / Update Data View", command = lambda: [clearframe(frame01,frame0),setDataView(v.get(),convert(),frame01,frame0,top2)],height = 1, width = 20, bd = 5, fg = 'black', bg ='gray')
                button1['font'] = sFont
                button1.pack()
                fb.pack()
                global rd
                snf = tk.Button(frame00, text = "Save Data to New File", command = lambda: [create_filtered(convert())],height = 1, width = 25, bd = 5, fg = 'black', bg ='gray')
                snf['font'] = sFont
                snf.pack(side = LEFT, padx = 10, pady = 5, expand = 1)
                
                def save_update():
                    # Update the data and save as new file
                    path = create_filtered(convert())+'.xlsx'
                    file.delete(0,'end')
                    file.insert(tk.END,path)
                    collist2 = list(convert().columns)
                    pred1['menu'].delete(0,'end')
                    for choice in collist2:
                        pred1['menu'].add_command(label = choice,command = tk._setit(predvar,choice))
                    top2.destroy()
                    
                snuf = tk.Button(frame00, text = "Save Data & Update NN Inputs", command = lambda: [save_update()],height = 1, width = 25, bd = 5, fg = 'black', bg ='gray')
                snuf['font'] = sFont
                
                
                def just_update():
                    # Update the data but do not save it to a new file
                    path = transfer(convert())
                    file.delete(0,'end')
                    file.insert(tk.END,path)
                    collist2 = list(convert().columns)
                    pred1['menu'].delete(0,'end')
                    for choice in collist2:
                        pred1['menu'].add_command(label = choice,command = tk._setit(predvar,choice))
                    top2.destroy()
                    
                uf = tk.Button(frame00, text = "Update NN Inputs", command = lambda: [just_update()],height = 1, width = 25, bd = 5, fg = 'black', bg ='gray')
                uf['font'] = sFont
                uf.pack(side = LEFT, padx = 80, pady = 5, expand =1)
                snuf.pack(side = LEFT, padx = 10, pady = 5, expand = 1)
                
            def setDataView(view, data,tlevel,slevel,vtlevel):
                # Configure settings for the selected data view option on the data exploration page
                collist = list(data.columns)
                if view == 0:
                    
                    summ = data.describe()
                    summ = summ.reindex(index=summ.index[::-1])
    
                    cols = list(summ.columns)
                    global tree
                    global sb
                    tree = ttk.Treeview(tlevel, selectmode = 'extended')
                    tree.pack()
                    sb = ttk.Scrollbar(tlevel, orient = "horizontal",command = tree.xview)
                    sb.pack(fill = X)
                    tree.configure(xscrollcommand = sb.set)
                    tree["columns"] = cols
                    for i in cols:
                        tree.column(i, anchor="w")
                        tree.heading(i, text=i, anchor='w')
    
                    for index, row in summ.iterrows():
                        tree.insert("",0,text=index,values=list(row))

                        
                elif view == 1:
                    ax2.clear()
                    corr = data.corr()
                    sns.heatmap(corr, annot = True, ax = ax2, cbar = False,cmap=sns.diverging_palette(20, 220, n=200))
                    canvas = FigureCanvasTkAgg(f3,slevel)
                    canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True)
                
                elif view == 2:
                    
                    lbl = tk.Label(tlevel, text = "X-Axis:",font = ('Helvetica', 14, 'bold'))
                    lbl.pack(side = LEFT, padx = 5)
                    xvar = StringVar()
                    xax = tk.OptionMenu(tlevel,xvar,value =None)
                    xvar.set(collist[0])
                    xax.pack(side= LEFT)
                    
    
                    lbl2 = tk.Label(tlevel, text = "Y-Axis:",font = ('Helvetica', 14, 'bold'))
                    lbl2.pack(side = LEFT, padx = 5)
                    yvar = StringVar()
                    yax = tk.OptionMenu(tlevel,yvar,value =None)
                    yvar.set(collist[0])
                    yax.pack(side = LEFT)
                    
                    def update(level):
                        
                        global frame
                        global rd
                        global fb
                        rd = fb.getdf()
                        try:
                            data = rd[0]
                            collist3 = data.columns
                            xax['menu'].delete(0,'end')
                            yax['menu'].delete(0,'end')
                            for choice in collist3:
                                xax['menu'].add_command(label = choice,command = tk._setit(xvar,choice))
                                yax['menu'].add_command(label = choice,command = tk._setit(yvar,choice))
                            
                        except Exception:
                            data = frame
                            
                        ax2.clear()
                        ax2.scatter(data[xvar.get()],data[yvar.get()])
                        ax2.set_xlabel(xvar.get())
                        ax2.set_ylabel(yvar.get())
                        mplcursors.cursor(ax2, hover=True)
                        canvas = FigureCanvasTkAgg(f3,level)
                        f3.canvas.draw()
                        canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True, padx = 10, pady = 10)

                    upd = tk.Button(tlevel,text = "Update Graph", command = lambda: [cleargraph(slevel),update(slevel)],height = 1, width = 20, bd = 5, fg = 'black', bg ='gray')
                    upd['font'] = sFont
                    upd.pack(side = TOP, padx = 30)
                    
                    for choice in collist:
                        xax['menu'].add_command(label = choice,command = tk._setit(xvar,choice))
                        yax['menu'].add_command(label = choice,command = tk._setit(yvar,choice))
                    
            def clearframe(tlevel,slevel):
                # Clear the data exploration page
                for widget in tlevel.winfo_children():
                    widget.destroy()
                cleargraph(slevel)

            def cleargraph(slevel):
                # Clear the data view graphic
                for widget in slevel.winfo_children():
                    widget.destroy()
            
            frame25 = tk.Frame(self)
            frame25.pack(fill=X)
                
            vis = tk.Button(frame1, text = "Data Exploration", command = create_top, bd = 2, fg = 'black', bg ='gray')
            vis.pack_forget()
            
                
            label1 = tk.Label(frame25, text = "Variable to Predict:",font = ('Helvetica', 14, 'bold'))
            fo = font.Font(label1, label1.cget("font"))
            label1.pack(side = LEFT, pady = 5, padx = 5)
            label1.configure(font=fo)
            
            global predvar
            predvar = StringVar()
            pred1 = tk.OptionMenu(frame25,predvar,value = None)
            predvar.set("Select a Data File First")
            pred1.pack(side = LEFT)
            
            def visual(*args):
                vis.pack(side = LEFT, padx = 5, pady=5)
                
            predvar.trace('w',visual)

            
            frame3 = tk.Frame(self)
            frame3.pack(fill = X)
            
            label2 = tk.Label(frame3, text = "Fraction of Data to Test:")
            label2.pack(side = LEFT, padx = 5, pady = 5)
            label2.configure(font=fo)
            
            frac = tk.Entry(frame3,justify = 'center')
            frac.config(width = 19)
            frac.pack(side = LEFT,padx = 5,pady = 5)
            
            frame4 = tk.Frame(self)
            frame4.pack(fill = X)
            
            label3 = tk.Label(frame4, text = "Number of Layers:")
            label3.pack(side = LEFT,padx = 5, pady = 5)
            label3.configure(font=fo)
            
            layers = tk.Entry(frame4,justify = 'center')
            layers.config(width = 28)
            layers.pack(side = LEFT, padx = 5,pady = 5)
            
            frame5 = tk.Frame(self)
            frame5.pack(fill = X)
            
            label4 = tk.Label(frame5, text = "Number of Neurons:")
            label4.pack(side = LEFT, padx = 5, pady = 5)
            label4.configure(font=fo)
            
            neurons = tk.Entry(frame5,justify = 'center')
            neurons.config(width = 25)
            neurons.pack(side = LEFT, padx = 5, pady = 5)
            
            frame6 = tk.Frame(self)
            frame6.pack(fill=X)
            
            label = tk.Label(frame6, text = "Save Model for Later Use?")
            label.pack(side = LEFT,pady = 5, padx = 5)
            
            label.configure(font=fo)
    
            sm = tk.IntVar()
            sm.set(0)
    
            rb1 = tk.Radiobutton(frame6, text = "Yes", variable = sm, value = 1).pack(side =LEFT)
            rb2 = tk.Radiobutton(frame6, text = "No", variable = sm, value = 0).pack(side=LEFT) 
            
            frame65 = tk.Frame(self)
            frame65.pack(fill=X)
            
            label5 = tk.Label(frame65, text = "Show Advanced User Settings?")
            label5.pack(side = LEFT,pady = 5, padx = 5)
            
            label5.configure(font=fo)
            
            tkvar=StringVar()
            choices = {'relu','logistic','tanh','identity'}
            activ = tk.OptionMenu(frame25,tkvar,*choices)
            tkvar.set('relu')
            label55 = tk.Label(frame25, text = 'Activation Function:')
            label55.configure(font=fo)
            
            label6 = tk.Label(frame3, text = 'Regularization:')
            label6.configure(font=fo)
            regs = tk.Entry(frame3, justify = 'center')
            regs.insert(0,.0001)
            regs.config(width = 25)
    
            label7 = tk.Label(frame4,text = 'Tolerance:')
            label7.configure(font=fo)
            tols = tk.Entry(frame4, justify = 'center')
            tols.insert(0,.00001)
            tols.config(width=25)
            
            label8 = tk.Label(frame5,text = 'Max Iterations:')
            label8.configure(font=fo)
            iters = tk.Entry(frame5, justify = 'center')
            iters.insert(0,50000)
            iters.config(width=25)
    
            label55.pack_forget()
            activ.pack_forget()
            regs.pack_forget()
            label6.pack_forget()
            label7.pack_forget()
            tols.pack_forget()
            label8.pack_forget()
            iters.pack_forget()
    
            
            def hide_opts():
                regs.pack_forget()
                label6.pack_forget()
                label55.pack_forget()
                activ.pack_forget()
                label7.pack_forget()
                tols.pack_forget()
                label8.pack_forget()
                iters.pack_forget()
                
            def show_opts():
                activ.pack(side = RIGHT,padx = 5,pady = 5)
                label55.pack(side = RIGHT,padx = 5,pady = 5)
                regs.pack(side = RIGHT,padx = 5,pady = 5)
                label6.pack(side = RIGHT,padx = 5,pady = 5)
                tols.pack(side = RIGHT,padx = 5,pady = 5)
                label7.pack(side = RIGHT,padx = 5,pady = 5)
                iters.pack(side = RIGHT,padx = 5,pady = 5)
                label8.pack(side = RIGHT,padx = 5,pady = 5)
                
            def defaults():
                tkvar.set('relu')
                regs.delete(0,'end')
                tols.delete(0,'end')
                iters.delete(0,'end')
                regs.insert(0,.0001)
                tols.insert(0,.00001)
                iters.insert(0,50000)
                
            rb3 = tk.Button(frame65, text = "Yes", command = show_opts).pack(side =LEFT)
            rb4 = tk.Button(frame65, text = "No", command = hide_opts).pack(side=LEFT)
            rb5 = tk.Button(frame65, text = "Reset", command = defaults).pack(side=LEFT)
            
            frame7 = tk.Frame(self)
            frame7.pack(fill=X)
            
            def NN():
                # Train the Neural Network
                progvar.set('')
                data = str(file.get())               
                if data == '':
                    messagebox.showerror('File Not Found Error','Please select an input file')
                    return
                yname = str(predvar.get())
                if yname == 'Choose Variable':
                    messagebox.showerror('Variable to Predict Error','Please select the variable you would like to train the neural network to predict')
                    return
                try:
                    testsize = float(frac.get())
                    if testsize >= 1 or testsize <= 0:
                        raise Exception('value')
                except Exception:       
                    messagebox.showerror('Test Size Error','Fraction of Data to Test must be a decimal between 0 and 1')
                    return
                try:
                    numlayers = int(layers.get())
                    if numlayers <= 0:
                        raise Exception('value')
                except Exception:
                    messagebox.showerror('Number of Layers Error','Number of Layers must be a positive integer')
                    return
                try:
                    layersize = int(neurons.get())
                    if layersize <= 0:
                        raise Exception('value')
                except Exception:
                    messagebox.showerror('Number of Neurons Error','Number of Neurons must be a positive integer')
                    return
                
                try:
                    iters1 = int(iters.get())
                    if iters1 <= 0:
                        raise Exception('value')
                except Exception:
                    messagebox.showerror('Iterations Error','Number of Iterations must be a positive integer')
                    return
                
                try:
                    tols1 = float(tols.get())
                except Exception:
                    messagebox.showerror('Tolerance Error','Tolerance Value must be numeric')
                    return
                
                try:
                    regs1 = float(regs.get())
                except Exception:
                    messagebox.showerror('Regularization Error','Regularization Value must be numeric')
                    return
                
                activ = str(tkvar.get())
                
                try:
                    edata = pd.read_excel(data)
        
                except Exception:
                    edata = pd.read_csv(data, sep=",")
    
                ecols = pd.DataFrame(data = edata.columns, columns = ['Columns'])
                is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
                types = pd.DataFrame(data = is_number(edata.dtypes), columns = ['Numeric'])
                
                efilt = ecols.merge(types, left_index = True, right_index = True)
                efilt = efilt[efilt['Numeric']==False]
                vals = efilt['Columns'].tolist()
                vals = 'The Data Column(s) '+' and '.join(vals) + ' contain(s) non-numeric data. This data cannot be used to train a multi-layer perceptron for regression purposes.'

                try:                   
                    if not efilt.empty:
                        raise Exception('value')
                    elif efilt.empty:
                        pass
                except Exception:
                    messagebox.showerror('Data Error',vals)
                    return
                
                global actual
                global predicted
                
                actual = "Actual"+" "+yname
                predicted = "Predicted"+" "+yname
        
                global train_score
                global test_score
                global train_data
                global test_data
                global avetrainerr
                global avetesterr
                global avedifftrain
                global avedifftest
                
                             
                progvar.set('Training in Progress...')
                progress.configure(foreground = 'red')
                self.update_idletasks()
    
                train_score,test_score,train_data,test_data,avetrainerr,avetesterr,avedifftrain,avedifftest = MLPRegress(sm.get(),data,yname,testsize,numlayers,layersize,activ,tols1,regs1,iters1)
         
  
                
                trainerr.config(text = "Average Train Error = "+str(round(avetrainerr,3)*100)+"%")
                testerr.config(text = "Average Test Error = "+str(round(avetesterr,3)*100)+"%")
                progvar.set('Training Complete!')
                progress.config(foreground = 'green')
                b.clear()
                
                b.set_xlabel(predicted)
                b.set_ylabel(actual)
                b.set_title("Test Dataset Performance")
                b.scatter(test_data[predicted],test_data[actual])
                mplcursors.cursor(b, hover=True)
    
        
                locator = MaxNLocator(prune = 'both', nbins = 20)
                b.xaxis.set_major_locator(locator)
                b.tick_params(axis = 'x', labelsize = 'small', rotation =  60)
    
                f2.canvas.draw() 
                                
            train=tk.Button(frame7,text="Train / Test",command=lambda: [NN()],height = 2, width = 20, bd = 5, fg = 'black', bg ='gray')
            train['font'] = myFont
            train.pack(side=LEFT,padx = 5,pady = 5)
            
            export=tk.Button(frame7,text="Export Report to Excel",command=lambda: [reports()],height = 2, width = 20, bd = 5, fg = 'black', bg ='gray')
            export['font']=myFont
            export.pack(side = RIGHT,padx = 5,pady = 5)
            def reports():
                try:    
                    create_report(train_data,test_data,train_score,test_score,avetrainerr,avetesterr,avedifftrain,avedifftest)
                except Exception:
                    messagebox.showerror('Error','No Report to Export, be sure to train the neural network first')
            frame8 = tk.Frame(self)
            frame8.pack(fill=X)
            
            trainerr = tk.Label(frame8,text = "Average Train Error  = ")
            trainerr.pack(side = LEFT,padx = 100,pady = 5)
            
            testerr = tk.Label(frame8,text = "Average Test Error  = ")
            testerr.pack(side=RIGHT,padx = 100,pady = 5)
            
            frame9 = tk.Frame(self)
            frame9.pack(fill=X)
            
            progvar = StringVar()
            progress = tk.Label(frame9,textvariable = progvar,font = ('Helvetica', 14, 'bold'))
            progress.pack()
                
            canvas = FigureCanvasTkAgg(f2,self)
            canvas.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True)
        frame = create(self) 
        

app = MLPlatform()

app.geometry("750x750")
try:
    app.mainloop()
finally:
    try:
        os.remove('temp_datafile_autodelete_on_exit.xlsx')
    except Exception:
        pass










