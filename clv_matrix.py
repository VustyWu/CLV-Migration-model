import numpy as np
import pandas as pd

class clv_matrix:
    def __init__(self, df, n, mode = 'outlier'):
        """
        Initializes the clv_matrix with a DataFrame and an integer defining the number of periods (n).
        
        Args:
        df (DataFrame): Input DataFrame containing customer data across multiple columns.
        n (int): Number of periods for calculation.
        """
        self.mode = mode
        self.n = n
        self.long = 2 ** self.n - 1  # Total number of calculations is 2^n - 1
        self.df = df
        self.bound = np.insert(np.cumsum([2 ** (self.n - i) for i in range(1, self.n)]), 0, 0)
        
    def df_split(self):
        """
        Splits the main DataFrame into rule, retention, and average DataFrames.
        
        Returns:
        tuple: Contains DataFrames for rules, retention, and calculated averages.
        """
        rule_df = self.df.iloc[:, :self.n + 2]
        df_rest = self.df.iloc[:, self.n + 2:]

        assert df_rest.shape[1] == 2 * self.n, "DataFrame does not have the expected number of columns."
        
        retention_df = df_rest.iloc[:, 0::2].copy()
        contribution_df = df_rest.iloc[:, 1::2].copy()
        average_df = pd.DataFrame(np.zeros_like(contribution_df.values), columns=contribution_df.columns)

        for i in range(self.n):
            for start in range(0, self.long, 2**(self.n - i)):
                end = min(start + 2**(self.n - i), self.long)
                if contribution_df.iloc[start:end, i].sum() != 0:
                    average = contribution_df.iloc[start:end, i].sum() / retention_df.iloc[start:end, i].sum()
                    subset_end = min(start + 2**(self.n - i - 1), self.long)
                    average_df.iloc[start:subset_end, i] = average

        self.rule = rule_df
        self.retention = retention_df
        self.contribution = average_df
        return rule_df, retention_df, average_df

    def retention_rate_matrix(self):
        """
        Calculates the retention rate matrix from the retention DataFrame by ensuring that the rate calculations
        adhere to the bounds specified by 'count' and 'ceiling'.

        Returns:
        tuple: Number DataFrame and Rate DataFrame derived from retention data.
        """
        number_df = pd.DataFrame(np.zeros_like(self.retention.values), columns=self.retention.columns)
        rate_df = pd.DataFrame(np.zeros_like(self.retention.values), columns=self.retention.columns)

        # Fill the last column of number_df based on the parity of the index
        for index, row in self.retention.iterrows():
            if index % 2 != 0:  # Odd index, calculate the mean of non-zero entries
                mean_value = row[row != 0].mean() if (row != 0).any() else 0
                number_df.iloc[index, -1] = mean_value
            else:  # Even index, take the last column directly
                number_df.iloc[index, -1] = row[-1]

        # Compute sums and rates for each subsequent column
        for step in range(1, self.n):
            index_num = 2 ** step
            ceiling = 2 ** (self.n + 1 - step) - 2  # Define the ceiling for the rate calculations
            count = 0  # Initialize count to keep track of index processing within ceiling limit

            for odd_index in range(0, self.long, index_num):
                even_index = odd_index + 2**(step - 1)
                if even_index < self.long and count < ceiling:
                    sum_value = number_df.iloc[odd_index, -step] + number_df.iloc[even_index, -step]
                    number_df.iloc[odd_index, -(step + 1)] = sum_value

                    if sum_value != 0:  # Avoid division by zero
                        rate1 = number_df.iloc[odd_index, -step] / sum_value
                        #rate2 = number_df.iloc[even_index, -step] / sum_value
                        rate_df.iloc[odd_index, -step] = rate1
                        #rate_df.iloc[even_index, -step] = rate2

                    count += 2  # Update count for each pair processed
                    
        if self.mode == 'outlier':
            rate_df = self.outlier_remedy(rate_df)



        # Iterate only over even rows (odd indices) starting from the second row
        for i in range(1, len(rate_df), 2):  # Start from index 1 and step by 2
            # Ensure the previous row index is within bounds
            if i > 0:
                # Set current even row (odd index) to 1 minus the previous odd row (even index) value
                rate_df.iloc[i, -1] = 1 - rate_df.iloc[i - 1, -1]

        
        # Assign special values based on bounds to the rate DataFrame
        step = self.n
        for bound_index in self.bound:
            rate_df.iloc[bound_index, -step] = 'S'  # Set 'S' at the specified bounds
            step -= 1
        
        self.number = number_df

        return number_df, rate_df


    def contribution_value_matrix(self):
        """
        Calculates a ratio matrix from the contribution DataFrame, considering only non-zero values.
        
        Returns:
        DataFrame: Ratio DataFrame computed from non-zero contribution values.
        """
        ratio_df = pd.DataFrame(np.zeros_like(self.contribution.values), columns=self.contribution.columns)
        self.contribution.replace(0, np.nan, inplace=True)

        for i in range(1, self.n):
            col_name = self.contribution.columns[i]
            ratio_df[col_name] = self.contribution.apply(
                lambda row: row[i] / row[:i].dropna().iloc[-1] if not pd.isna(row[i]) and row[:i].dropna().any() else np.nan,
                axis=1
            )
        if self.mode == 'outlier':
            ratio_df = self.outlier_remedy(ratio_df)

        return ratio_df.replace(0, np.nan)
    
    def clv_year_matrix(self):
        clv_df = pd.DataFrame(np.zeros_like(self.retention.values), columns=self.retention.columns)
        for j in range(self.n):
            gap = 2 ** (self.n - j - 1)
            for row in range(0, self.long, gap):
                tmp_val = self.number.iloc[row,j] * self.contribution.iloc[row,j]
                clv_df.iloc[row,j] = tmp_val
                
        clv_df = clv_df.fillna(0)
        clv_df = clv_df.applymap(lambda x: 0 if abs(x) < 1e-9 else x)

        return clv_df
    
    def outlier_remedy(self, df):
        # Iterate over each column in DataFrame
        for column in df.columns:
            # Attempt to convert column to numeric, coerce errors to NaN
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Replace all NaN values with 0
        #df.fillna(0, inplace=True)
        column_dict = {}

        # Iterate over each column again to process non-NaN values
        for column in df.columns:
            # Filter to include only even indices (odd rows)
            filtered_df = df.iloc[df.index % 2 == 0]

            # Exclude zeros and check if there are any non-NaN values in the column
            non_zero_df = filtered_df[filtered_df[column] != 0]

            if not non_zero_df[column].isna().all():

                # Populate the dictionary with value-index pairs
                for index, value in non_zero_df[column].dropna().items():
                    column_index = df.columns.get_loc(column)
                    column_dict[value] = [index, column_index]
                    
        # Extract the keys as a list of floats
        values = list(column_dict.keys())

        # Calculate IQR to determine outliers
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = {value: column_dict[value] for value in values if value < lower_bound or value > upper_bound}
        # Extract the keys as a list of floats
        outlier_values = list(outliers.keys())

        # Apply log transformation only to outliers
        transformed_values = {
            original: ((np.log10(original - upper_bound + 1) + upper_bound if original > upper_bound else
             -np.log10(lower_bound - original + 1) + lower_bound if original < lower_bound else
             original))
            for original in outliers.keys()
        }

        # Iterate over each cell in the DataFrame
        for (row_idx, col), value in np.ndenumerate(df):
            if value in list(outlier_values):
                # Update the DataFrame only if the value is a recognized outlier
                df.iloc[row_idx, col] = transformed_values[value]


        return df



        
                
        
