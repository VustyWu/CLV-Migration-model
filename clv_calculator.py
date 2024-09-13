import numpy as np
import pandas as pd

class CLVCalculator:
    """
    A class to calculate Customer Lifetime Value (CLV) using various data frames
    that inform the calculation through rates, contributions, retentions, etc.
    """

    def __init__(self, rule_df, retention_df, rate_df, contribution_df, ratio_df, clv_df, n, predict_n, 
                 future_retention_rate, future_retention_ratio):
        """
        Initializes the CLVCalculator with required data and parameters.

        Args:
        rule_df (DataFrame): DataFrame containing rules for calculating weights.
        retention_df (DataFrame): DataFrame containing retention rates.
        rate_df (DataFrame): DataFrame containing rates.
        contribution_df (DataFrame): DataFrame containing contributions.
        ratio_df (DataFrame): DataFrame containing ratios.
        n (int): The number of periods for calculating CLV.
        predict_n (int): Number of future periods to predict.
        future_retention (list): Future retention rates.
        """
        
        self.constant_n = n
        self.future_retention_rate = future_retention_rate
        self.future_retention_ratio = future_retention_ratio
        self.n = n
        self.predict = predict_n
        self.long = 2 ** self.n - 1
        self.rule = rule_df
        self.contribution = contribution_df.fillna(0)
        self.clv = contribution_df.fillna(0)
        self.retention = retention_df
        self.ratio = ratio_df
        self.rate = rate_df
        self.clv = clv_df
        self.index = np.arange(0, n)
        self.bound = np.insert(np.cumsum([2 ** (self.n - i) for i in range(1, self.n)]), 0, 0)

    def get_row_indices(self, start_bound, end_bound, gap):
        """Generate indices for rows within specified bounds with a given gap."""
        return list(range(start_bound, end_bound, gap))

    def get_n_col(self, current_iter, target_iter):
        """Calculate the column index offset from current to target iteration."""
        return -(target_iter - current_iter + 1)

    def get_weights(self, k):
        """
        Calculate the weights for each value in the number DataFrame based on the rule DataFrame.

        Args:
        number_df (DataFrame): The DataFrame from which to calculate weights.

        Returns:
        list of lists: The weights for each value.
        """
        assert len(self.retention) == 2 ** (self.n + k - 1) - 1
        result_list = [[] for _ in range(self.long)]
        rev_bound = np.insert(self.bound[::-1], 0, self.long)
        start_bound = k - 1
        for j in range(start_bound, len(self.bound) - 1):
            current_iter = self.index[j]
            start_bound = rev_bound[j + 1]
            end_bound = rev_bound[j]
            upper_range = self.index[j + 1 : self.n + k - 1]
            #upper_range = self.index[j+1:j+1+self.n-j]
            current_bound = np.arange(start_bound, end_bound)


            for upper_index in upper_range:
                gap = 2 ** (upper_index - current_iter)
                row_indices = self.get_row_indices(rev_bound[upper_index+1], rev_bound[upper_index], gap)
                col_index = self.get_n_col(current_iter, upper_index)
                for i, current_row in enumerate(current_bound):
                    study_row = row_indices[i]
                    tmp_val = self.retention.iloc[study_row, col_index]
                    result_list[current_row].append(tmp_val)

                    
        for i in range(len(result_list)):
            sublist = result_list[i]
            total_value = sum(sublist)
            weights = [w / total_value for w in sublist]
            result_list[i] = weights

        return result_list

    def multiply_and_sum(self, list1, list2):
        """Multiply elements and sum products for corresponding sublists from two lists."""
        return [sum(i * j for i, j in zip(sublist1, sublist2)) for sublist1, sublist2 in zip(list1, list2)]

    def calculate_rate(self, k, weights):
        """
        Calculate the rate of change based on the retention data and future retention assumptions.
        Safely handles the indices for future_retention to avoid IndexError.

        Args:
        i (int): Index to handle specific iteration adjustments. Expected to start from 1.
        """
        result_list = [[] for _ in range(self.long)]
        rev_bound = np.insert(self.bound[::-1], 0, self.long)
        start_bound = k - 1
        for j in range(start_bound, len(self.bound) - 1):
            current_iter = self.index[j]
            start_bound = rev_bound[j + 1]
            end_bound = rev_bound[j]
            upper_range = self.index[j + 1 : self.n + k - 1]
            #upper_range = self.index[j+1:j+1+self.n-j]
            current_bound = np.arange(start_bound, end_bound)

            for upper_index in upper_range:
                gap = 2 ** (upper_index - current_iter)
                row_indices = self.get_row_indices(rev_bound[upper_index+1], rev_bound[upper_index], gap)
                col_index = self.get_n_col(current_iter, upper_index) + 1

                for i, current_row in enumerate(current_bound):
                    study_row = row_indices[i]
                    tmp_val = self.rate.iloc[study_row, col_index]
                    result_list[current_row].append(tmp_val)

        result = self.multiply_and_sum(result_list, weights)
        for idx, val in enumerate(result):
            if val == 0:
                safe_index = min(k - 1, len(self.future_retention_rate) - 1)
                result[idx] = self.future_retention_rate[safe_index]

        return result
    
    
    def calculate_cv(self, k, weights):

        """
        Calculate the customer value based on the ratio data and future retention assumptions.
        Safely handles the indices for future_retention to avoid IndexError.

        Args:
        i (int): Index to handle specific iteration adjustments. Expected to start from 1.
        """
        result_list = [[] for _ in range(self.long)]
        rev_bound = np.insert(self.bound[::-1], 0, self.long)
        start_bound = k - 1
        for j in range(start_bound, len(self.bound) - 1):
            current_iter = self.index[j]
            start_bound = rev_bound[j + 1]
            end_bound = rev_bound[j]
            upper_range = self.index[j + 1 : self.n + k - 1]
            #upper_range = self.index[j+1:j+1+self.n-j]
            current_bound = np.arange(start_bound, end_bound)

            for upper_index in upper_range:
                gap = 2 ** (upper_index - current_iter)
                row_indices = self.get_row_indices(rev_bound[upper_index+1], rev_bound[upper_index], gap)
                col_index = self.get_n_col(current_iter, upper_index) + 1  # Add 1 to shift from zero-based index if needed

                for index, current_row in enumerate(current_bound):
                    study_row = row_indices[index]
                    tmp_val = self.ratio.iloc[study_row, col_index]
                    result_list[current_row].append(tmp_val)

        result = self.multiply_and_sum(result_list, weights)
        for idx, val in enumerate(result):
            if val == 0:
                # Ensure i-1 is a valid index; otherwise use the last available value
                safe_index = min(k - 1, len(self.future_retention_ratio) - 1)
                result[idx] = self.future_retention_ratio[safe_index]

        return result
    
    def extend_matrix(self, df):
        zeros_df = pd.DataFrame(0, index=np.arange(len(df)), columns=df.columns)  
        df = pd.concat([df, zeros_df]).sort_index(kind='merge')
        df.reset_index(drop=True, inplace=True)
        # Create a pandas Series with zeros for all columns
        new_row = pd.Series([0] * len(df.columns), index=df.columns)
        # Add the new row at the end of the DataFrame
        df.loc[len(df)] = new_row
        return df

    
    def get_future_matrix(self):
        """
        Extends and calculates future matrices for retention, rate, ratio, and contribution.
        Iteratively adds prediction data based on weights calculated for future periods.
        """
        # Copy the initial matrices to keep extending them with future data.
        future_retention = self.retention.copy()
        future_rate = self.rate.copy()
        future_ratio = self.ratio.copy()
        future_contribution = self.contribution.copy()
        future_clv = self.clv.copy()

        # Loop over the number of future periods to predict.
        for j in range(1, self.predict + 1):
            weights = self.get_weights(j)  # Calculate weights for this future period
            cv_column = self.calculate_cv(j, weights)  # Calculate customer value for this period
            rate_column = self.calculate_rate(j, weights)  # Calculate rate of change for this period
            name = f'未来第{j}年数据'  # Naming convention for new columns

            # Sanity checks to ensure consistent data lengths
            assert len(cv_column) == len(rate_column), "Mismatch in cv and rate column lengths."
            assert len(cv_column) == future_rate.shape[0], "Mismatch in data frame lengths."

            # Add new calculated data as columns to the matrices
            future_rate[name] = rate_column
            future_ratio[name] = cv_column

            # Extend matrices by appending zeros to accommodate new data
            future_retention = self.extend_matrix(future_retention)
            future_contribution = self.extend_matrix(future_contribution)
            future_clv = self.extend_matrix(future_clv)
            future_rate = self.extend_matrix(future_rate)
            future_ratio = self.extend_matrix(future_ratio)

            # Adjust future rates based on the previously calculated values
            for i in range(1, len(future_rate), 2):  # Iterate over even rows
                future_rate.iloc[i, -1] = 1 - future_rate.iloc[i-1, -1]

            # Calculate new retention and contribution values based on extended matrices
            new_column_length = len(future_retention)
            new_column = np.zeros(new_column_length)
            new_column2 = np.zeros(new_column_length)
            new_column3 = np.zeros(new_column_length)
            for i in range(0, len(future_rate), 2):
                if i == (len(future_rate) - 1):
                    break
                else:
                    new_column[i] = future_retention.iloc[i, -1] * future_rate.iloc[i, -1]
                    new_column[i + 1] = future_retention.iloc[i, -1] * future_rate.iloc[i + 1, -1]
                    tmp_val_cv = self.get_nearest_value(future_contribution.iloc[i])
                    new_column2[i] = tmp_val_cv * future_ratio.iloc[i, -1]
                    new_column2[i + 1] = tmp_val_cv

            # Update matrices with new calculated columns
            future_retention[name] = new_column
            future_contribution[name] = new_column2
            clv_column = self.get_clv(future_retention, future_contribution, new_column3)
            future_clv[name] = clv_column

            # Update calculation bounds and indices for next iteration
            self.bound = np.insert(np.cumsum([2 ** (self.n + j - i) for i in range(1, self.n + j)]), 0, 0)
            self.retention = future_retention
            self.contribution = future_contribution
            self.rate = future_rate
            self.ratio = future_ratio
            self.clv = future_clv
            self.long = len(future_rate)
            self.index = np.arange(0, self.n + j + 1)

        return future_retention, future_ratio, future_rate, future_contribution, future_clv

    
    def get_nearest_value(self, row):
        """
        Retrieve the nearest non-zero value from a pandas Series, searching from the end.
        """
        for value in reversed(row.values):  # Iterate from the end to start
            if value != 0:
                return value
        return 0  # Return zero if no non-zero values are found

    def calculate_ratio_product(self, row):
        """
        Calculate the product of non-zero values in a pandas Series.
        """
        nonzero_values = row[row != 0]  # Filter out zero values
        product = nonzero_values.product()  # Calculate product of nonzero values
        return product

    def get_clv(self, future_retention, future_contribution, new_column):
        """
        Compute the Customer Lifetime Value (CLV) by multiplying and summing specific matrix elements.
        """
        #total_sum = 0
        for i in range(len(future_retention)):
            if i % 2 == 0:  # Only calculate for even indices
                # total_sum += future_retention.iloc[i, -1] * future_contribution.iloc[i, -1]
                new_column[i] = future_retention.iloc[i, -1] * future_contribution.iloc[i, -1]

        #print(f'CLV: {total_sum}')
        return new_column
    
