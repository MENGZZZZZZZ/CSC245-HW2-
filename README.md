# CSC245-HW2-

# 1. Write a Pandas program to create and display a one-dimensional array-like object containing an array of data using Pandas module.
import pandas as pd
data = pd.Series([10, 20, 30, 40, 50])
print(data)

# 2. Write a Pandas program to convert a Panda module Series to Python list and it's type.
import pandas as pd
series_data = pd.Series([10, 20, 30, 40, 50])
list_data = series_data.tolist()
print(list_data)
print(type(list_data))

# 3. Write a Pandas program to add, subtract, multiple and divide two Pandas Series.
# Sample Series: [2, 4, 6, 8, 10], [1, 3, 5, 7, 9]
import pandas as pd
series1 = pd.Series([2, 4, 6, 8, 10])
series2 = pd.Series([1, 3, 5, 7, 9])
addition = series1 + series2
subtraction = series1 - series2
multiplication = series1 * series2
division = series1 / series2
print("Addition of two series:")
print(addition)
print("\nSubtraction of two series:")
print(subtraction)
print("\nMultiplication of two series:")
print(multiplication)
print("\nDivision of two series:")
print(division)

# 4. Write a Pandas program to compare the elements of the two Pandas Series.
# Sample Series: [2, 4, 6, 8, 10], [1, 3, 5, 7, 10]
import pandas as pd
series1 = pd.Series([2, 4, 6, 8, 10])
series2 = pd.Series([1, 3, 5, 7, 10])
gt = series1 > series2
lt = series1 < series2
eq = series1 == series2
print("Series1 greater than Series2:")
print(gt)
print("\nSeries1 less than Series2:")
print(lt)
print("\nSeries1 equal to Series2:")
print(eq)

# 5. Write a Pandas program to convert a dictionary to a Pandas series. 
import pandas as pd
dict_data = {'a': 100, 'b': 200, 'c': 300, 'd': 400, 'e': 800}
series_data = pd.Series(dict_data)
print(series_data)

# 6. Write a Pandas program to convert a NumPy array to a Pandas series.
import pandas as pd
import numpy as np
np_array = np.array([10, 20, 30, 40, 50])
series_from_np = pd.Series(np_array)
print(series_from_np)

# 7. Write a Pandas program to change the data type of given a column or a Series. 
import pandas as pd
original_series = pd.Series([100, 200, 'python', 300.12, 400])
numeric_series = pd.to_numeric(original_series, errors='coerce')
print(numeric_series)

# 8. Write a Pandas program to convert the first column of a DataFrame as a Series.  
import pandas as pd
original_dataframe = pd.DataFrame({
    'col1': [1, 2, 3, 4, 7, 11],
    'col2': [4, 5, 6, 9, 5, 0],
    'col3': [7, 5, 8, 12, 1, 11]
})
column_as_series = original_dataframe['col1']
print(column_as_series)
print(type(column_as_series))

# 9. Write a Pandas program to convert a given Series to an array. 
import pandas as pd
original_series = pd.Series([100, 200, 'python', 300.12, 400])
series_to_array = original_series.values
print(series_to_array)
print(type(series_to_array))

# 1.Write a Pandas program to create a dataframe from a dictionary and display it.
# Sample data: {'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]}
import pandas as pd
data = {
    'X': [78, 85, 96, 80, 86],
    'Y': [84, 94, 89, 83, 86],
    'Z': [86, 97, 96, 72, 83]
}
df = pd.DataFrame(data)
print(df)

# 2. Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
print(df)

# 3. Write a Pandas program to display a summary of the basic information about a specified DataFrame and its data.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
df.info()

# 4. Write a Pandas program to get the first 3 rows of a given DataFrame.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
first_three_rows = df.head(3)
print(first_three_rows)

# 5. Write a Pandas program to select the 'name' and 'score' columns from the following DataFrame.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
selected_columns = df[['name', 'score']]
print(selected_columns)

# 6. Write a Pandas program to select the specified columns and rows from a given data frame.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
selected_data = df.loc[['b', 'd', 'f', 'g'], ['score', 'qualify']]
print(selected_data)

# 7. Write a Pandas program to select the rows where the number of attempts in the examination is greater than 2.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
rows_with_more_attempts = df[df['attempts'] > 2]
print(rows_with_more_attempts)

# 8. Write a Pandas program to count the number of rows and columns of a DataFrame.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
number_of_rows = df.shape[0]
number_of_columns = df.shape[1]
print("Number of Rows:", number_of_rows)
print("Number of Columns:", number_of_columns)

# 9. Write a Pandas program to select the rows where the score is missing, i.e. is NaN.
import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
rows_with_missing_score = df[df['score'].isna()]
print(rows_with_missing_score)

# 10. Write a Pandas program to select the rows the score is between 15 and 20 (inclusive).

import pandas as pd
import numpy as np
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data, index=labels)
rows_with_scores_within_range = df[df['score'].between(15, 20)]
print(rows_with_scores_within_range)
























