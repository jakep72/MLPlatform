import pandas as pd
from datetime import date
from tkinter.filedialog import asksaveasfilename
from tkinter import messagebox

def transfer(data):
    
    filtered_data = data         
    filename = "temp_datafile_autodelete_on_exit.xlsx"
    writer = pd.ExcelWriter(filename, engine = 'xlsxwriter') 
    filtered_data.to_excel(writer,sheet_name = 'Filtered Data', index = False)
    workbook = writer.book
    worksheet = writer.sheets['Filtered Data']
    writer.save()

    return (filename)

def create_filtered(data):
    
    filtered_data = data          
    filename = asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                         ("All files", "*.*") )) 
    writer = pd.ExcelWriter(filename+'.xlsx', engine = 'xlsxwriter') 
    filtered_data.to_excel(writer,sheet_name = 'Filtered Data', index = False)
    workbook = writer.book
    worksheet = writer.sheets['Filtered Data']
    writer.save()

    return (filename)

def create_report(data1,data2,score1,score2,err1,err2,diff1,diff2):

    train_data = data1
    test_data = data2
    train_score = score1
    test_score = score2
    avetrainerror = err1
    avetesterror = err2
    avetraindiff = diff1
    avetestdiff = diff2
    
    trainstats = train_data.describe()
    teststats = test_data.describe()
    
    trainlen = train_data.shape[0]
    testlen = test_data.shape[0]
    trainwidth = train_data.shape[1]
    testwidth = test_data.shape[1]
    
    summ = {'Dataset':['Train','Test'],
            'Average Difference':[avetraindiff,avetestdiff],
            'Average Error':[avetrainerror,avetesterror],
            'R^2 Score':[train_score,test_score]}
    
    summary = pd.DataFrame(summ, columns = ['Dataset','Average Difference','Average Error','R^2 Score'])    
        
    filename = asksaveasfilename(filetypes=(("Excel files", "*.xlsx"),
                                                         ("All files", "*.*") )) 
    writer = pd.ExcelWriter(filename+'.xlsx', engine = 'xlsxwriter') 
    
    summary.to_excel(writer,sheet_name = 'Quick Summary', index = False)
    
    trainstats.to_excel(writer, sheet_name = 'Train Summary')
    teststats.to_excel(writer, sheet_name = 'Test Summary')
    train_data.to_excel(writer, sheet_name = 'Train Raw Data')
    test_data.to_excel(writer, sheet_name = 'Test Raw Data')

    
    workbook = writer.book
    worksheet = writer.sheets['Quick Summary']
    

    
    chart = workbook.add_chart({'type': 'scatter'})
    chart.add_series({
        'name':'Training Data',
        'categories': ['Train Raw Data',1,(trainwidth-2),trainlen,(trainwidth-2)],
        'values':['Train Raw Data',1,(trainwidth-3),trainlen,(trainwidth-3)]})
    chart.add_series({
        'name':'Testing Data',
        'categories': ['Test Raw Data',1,(testwidth-2),testlen,(testwidth-2)],
        'values':['Test Raw Data',1,(testwidth-3),testlen,(testwidth-3)]})
    chart.set_x_axis({'name':'Predicted'})
    chart.set_y_axis({'name':'Actual'})
    chart.set_size({'width': 720, 'height': 420})
    chart.set_title({'name':'Actual v Predicted'})
    
    worksheet.insert_chart('F1',chart)
    writer.save()