# To-churn-or-not-to-churn-from-a-service

churn dataset columns:
int or float 
{Account_Length, Vmail_Message, Day_Mins, Eve_Mins, Night_Mins, Intl_Mins, 
CustServ_Calls, Day_Calls, Day_Charge, Eve_Calls, Eve_Charge, Night_Calls, Night_Charge, Intl_Calls, Intl_Charge, Area_Code}

and object columns are 
{State, Churn, Intl_Plan, Vmail_Plan, Phone}

The numerical columns have the mean values as: [101.1, 8.1, 179.8, 201.0, 201.0, 10.23, 1.56, 100.43, 30.56, 100.11, 17.10, 100.10, 9.03, 4.50]

We see that the values are ranging from 1.56 to 2001. So standardization is useful for model building. 
The data has no missing values but all the colums have many missing values. 
The data and the code based on a course from the Data camp but I did substantial imporovement on the coding. 

