import pandas as pd





raw_csv_data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/MLData/DS_ML_Data/Part_8_Case_Study/S58_L412/Absenteeism-data.csv')
df = raw_csv_data.copy()
pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = df.drop(['ID'], axis=1)
RFAbsense = df['Reason for Absence']
# print(sorted(RFAbsense))
reason_columns = pd.get_dummies(RFAbsense)

reason_columns['check'] = reason_columns.sum(axis=1)
# print(reason_columns['check'].sum(axis=0))
# print(reason_columns['check'].unique())
reason_columns= reason_columns.drop(['check'], axis = 1)

reason_columns = pd.get_dummies(RFAbsense, drop_first=True)##this avoid multicolinearity issues
df = df.drop(['Reason for Absence'], axis =1)

reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

list_months =[]
for i in range(df.shape[0]):
    list_months.append(df['Date'][i].month)

df['Month Value'] = list_months

def date_to_weekday(date_value):
    return date_value.weekday()

df['Day of the Week'] = df['Date'].apply(date_to_weekday)
df = df.drop(['Date'], axis =1)

df['Education'] = df['Education'].map({1:0 , 2:1, 3:1, 4:1})

df_preprocessed=df.copy()


df_preprocessed.to_csv('/Users/tannerbraithwaite/github/python_scripts/MLData/absenteeismdata_preprocessed.csv', index=False)
