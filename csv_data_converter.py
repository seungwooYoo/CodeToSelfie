import pandas as pd

if __name__ == "__main__":
    '''
    Main function for converting kaggle wallmart competition
    Data type : 
    TripType - categorized as 38 trips
    '''
    data_df = pd.read_csv('train.csv', 
                          names=['TripType', 'VisitNumber',
                                 'Weekday', 'Upc', 'ScanCount', 
                                 'DepartmentDescription', 'FinelineNumber'], 
                          header=0)
    

    print data_df.values[0]



