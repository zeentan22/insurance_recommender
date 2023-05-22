import pandas as pd
import numpy as np
import functools 

class InsuranceRecommender:

    def __init__(self, linguistic: dict, full_input: dict, full_data: pd.DataFrame):
        self.linguistic = linguistic
        self.user_input = full_input["user_input"]
        self.user_demographics = full_input["user_demographics"]
        self.df = full_data
        self.policy_rankings = None
        

    def filter_based_on_demographics(self, constraints = None) -> None:
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


    def generate_grey_relational_coefficient(self, df : pd.DataFrame, best_policy_val = 100, rho = 0.1) -> pd.DataFrame:
        d = {}
        policies = df.index.tolist()
        # print(policies)
        print(df)
        for column in df:
            temp_df = df[[column]]
            best_policy = np.repeat(100,len(df))
            coeff_list = []

            # same as the for loop below
            min_ij = np.amin(best_policy - temp_df[column].to_numpy())
            max_ij = np.amax(best_policy - temp_df[column].to_numpy())

            # to perform optimisation, numpy vectorization is used instead of iterating through the dataframe...
            # same as the for loop below
            min_ij_np = np.repeat(min_ij, len(df))
            max_ij_np = np.repeat(max_ij, len(df))

            rho_np = np.repeat(rho, len(df))
            dist_ij = best_policy - temp_df[column].to_numpy()
            coeff_list2  = (min_ij_np + rho_np * max_ij_np) / (dist_ij + rho_np * max_ij_np).tolist()
            
            d[column] = coeff_list2
            corr_df = pd.DataFrame(data = d, index = policies)
            # print(corr_df)
        return corr_df

    def grey_relational_grade(self, weight: dict, grey_relational_coefficient: pd.DataFrame) -> dict:
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
        print(sum_grade)
            
        return d

    def pipeline(self, weight: dict, recommendations : int,  best_policy_value : int, rho : float) -> list:
        self.filter_based_on_demographics()
        corr_df = self.generate_grey_relational_coefficient(self.df, best_policy_value, rho)
        grade = self.grey_relational_grade(weight, corr_df)
        sorted_policies = {k: v for k, v in sorted(grade.items(), key=lambda item: item[1], reverse = True)}
        if recommendations < 0:
            raise ValueError(f"Reccomendations cannot be negative")
        if recommendations > len(sorted_policies):
            raise IndexError(f"Recommendations exceed number of policies! Recommendations should be less than {len(sorted_policies)}" )
        policies = list(sorted_policies.keys())
        self.policy_rankings = sorted_policies
        return policies[0:recommendations]
    


    def run(self, recommendations = 1, best_policy_value = 100, rho = 0.1) -> list:
        weights2 = self.preference_modeling(self.user_input, self.linguistic)
        policies = self.pipeline(weights2, recommendations, best_policy_value, rho)
        return policies

    def __repr__(self):
        output = "Recommender system \n"
        for policy, value in self.policy_rankings.items():
            output = output + f"{policy} : {round(value,5)} \n"
        return output