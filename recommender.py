import pandas as pd
import numpy as np
import functools 

class InsuranceRecommender:

    def __init__(self, linguistic, user_input, full_data):
        self.linguistic = linguistic
        self.user_input = user_input[1]
        self.user_demographics = user_input[0]
        self.df = full_data
        pass

    
    def filter_based_on_demographics(self, constraints = None):
        user_age = self.user_demographics["Age"]
        self.df = self.df[self.df["Entry Age"] < user_age]
        self.df = self.df.drop("Entry Age", axis = 1)


    def calculate(self, IFNs : list) -> float:
        return IFNs[0] + IFNs[2] * (IFNs[0] / (IFNs[0] + IFNs[1]))


    def preference_modeling(self, user_pref : dict, linguistic: dict) -> dict:
        weights = {}
        preferences = list(user_pref.keys())
        sum_pref = 0
        for pref in preferences:
            importance = user_pref[pref]
            IFNs = linguistic[importance]
            numerator = self.calculate(IFNs)
            weights[pref] = numerator
            sum_pref += numerator
        for pref in preferences:
            weights[pref] = weights[pref] / sum_pref
        return weights


    def generate_grey_relational_coefficient(self, df, best_policy_val = 100, rho = 0.1):
        d = {}
        policies = df.index.tolist()
        # print(policies)
        for column in df:
            temp_df = df[[column]]
            coeff_list = []
            min_ij, max_ij = 100 , 0
            for policy in policies:
                policy_i = temp_df[column][policy]
                for row in temp_df.itertuples():
                    if row.Index == policy:
                        continue
                    policy_j = getattr(row,column)
                    dist_ij = abs(best_policy_val - policy_j)
                    if dist_ij > max_ij:
                        max_ij = dist_ij
                    if dist_ij < min_ij:
                        min_ij = dist_ij

            for row in temp_df.itertuples():
                policy_i = getattr(row,column)
                dist_ij = best_policy_val - policy_i
                coeff = (min_ij + rho * max_ij) / (dist_ij + rho * max_ij)
                coeff_list.append(coeff)

            d[column] = coeff_list

            corr_df = pd.DataFrame(data = d, index = policies)
            
        return corr_df

    def grey_relational_grade(self, weight, grey_relational_coefficient):
        grade = 0
        policies = grey_relational_coefficient.index.tolist()
        d = {}
        sum_grade = 0
        for policy in policies:
            grade = 0
            for column in self.df:
                grade += weight[column] * grey_relational_coefficient[column][policy]  
            sum_grade += grade
            d[policy] = grade
            
        for policy in policies:
            d[policy] = d[policy] / sum_grade
            
        return d

    def pipeline(self, weight, recommendations,  best_policy_value, rho):
        self.filter_based_on_demographics()
        corr_df = self.generate_grey_relational_coefficient(self.df, best_policy_value, rho)
        grade = self.grey_relational_grade(weight, corr_df)
        sorted_policies = {k: v for k, v in sorted(grade.items(), key=lambda item: item[1], reverse = True)}
        if recommendations > len(sorted_policies):
            raise IndexError(f"Recommendations exceed number of policies! Recommendations should be less than {len(sorted_policies)}" )
        policies = list(sorted_policies.keys())
        return policies[0:recommendations]

    def run(self, recommendations = 2, best_policy_value = 100, rho = 0.1):
        weights2 = self.preference_modeling(self.user_input, self.linguistic)
        policies = self.pipeline(weights2, recommendations, best_policy_value, rho)
        return policies
