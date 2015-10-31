import pandas as pd
import math

if __name__ == "__main__":
    '''
    Main function for converting kaggle wallmart competition
    Data type : 
    0 / TripType - categorized as 38 trips 
               representing the type of shopping trip the customer made
               (should be predicted in test datasets)
               999 : other category (may be mixtures..?) 
    1 / VisitNumber - an id corresponding to a single trip by a single customer
    2 / Weekday - The weekday of the trip
    3 / Upc - the UPC number of the product purchased (barcode - 12 numeric characters)
    4 / ScanCount - number of the purchased item (-1 : returned)
    5 / DepartmentDescription - Item's department
    6 / FinelineNumber - refined category (created by Wallmart) - May correlated with department description
    '''
    data_df = pd.read_csv('train.csv', 
                          names=['TripType', 'VisitNumber',
                                 'Weekday', 'Upc', 'ScanCount', 
                                 'DepartmentDescription', 'FinelineNumber'], 
                          header=0)

    '''
    Preprocessing data
    1. Required to convert text to num
    - Weekday : Monday, ... , Sunday 
    - DepartmentDescription : 
    2. Upc : may required to hashing trick or dimensional reduction 
             since the number is too large
    3. Consider missing data - missing values should be considered 
                             - Let's firstly remove them (?)
    '''
    
    missing_removed = []
    for values in data_df.values:
        if math.isnan(values[0]):
            continue
        elif math.isnan(values[1]):
            continue
        elif isinstance(values[2], basestring) == False: 
            continue
        elif math.isnan(values[3]):
            continue
        elif math.isnan(values[4]):
            continue
        elif isinstance(values[5], basestring) == False: 
            continue
        elif math.isnan(values[6]):
            continue
        missing_removed.append(values)
    
    '''
    Training size reduced to 642925 from 647054 (4129 missed)
    '''

    weekday_table = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 
                     'Thursday':3, 'Friday':4, 'Saturday':5, 
                     'Sunday':6}

    for values in missing_removed:
        new_week_value = weekday_table[values[2]]
        values[2] = new_week_value

    '''
    Make a unique table
    '''
    department_table = {}
    unique_department_id = 0
    for values in missing_removed:
        cur_department_val = values[5]
        if department_table.has_key(cur_department_val) == False:
            department_table[cur_department_val] = unique_department_id
            unique_department_id += 1

    print 'Total number of department id {0}'.format(unique_department_id)
    '''
    68 unique department classs 
    '''
