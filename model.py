import pandas as pd


linguistic = {
    "Very Important" : [0.9,0.05,0.05],
    "Important" : [0.75, 0.2, 0.05],
    "Medium" : [0.5, 0.4, 0.1],
    "Unimportant" : [0.25, 0.6, 0.15],
    "Very unimportant": [0.1, 0.8, 0.1]
}


d = {"Low_Premium": [100,80,72,92,86], 
     "Flexibility": [0,100,0,75,50],
      "Tax_benefits": [60,80,86,64,70],
      "Benefits_death": [66,85,86,70,74],
      "Benefits_survival": [64,90,88,74,80],
      "Good_customer_service": [80,80,80,80,80],
      "Bonus": [80,100,90,90,100],
      "Special_Schemes": [60,60,80,0,0],
      "Riders":
      [100,0,0,100,100]}

df = pd.DataFrame(data = d, index = ["P1",'P2','P4','P6','P7'])
df.to_csv("performance.csv")

user_input = {
    "Low_Premium" : "Very Important",
    "Flexibility" : "Unimportant",
    "Tax_benefits" : "Medium",
    "Benefits_death" : "Very Important",
    "Benefits_survival" : "Important",
    "Good_customer_service" : "Unimportant",
    "Bonus" : "Important",
    "Special_Schemes": "Very unimportant",
    "Riders" : "Medium"
}

weights = {
    "Low_Premium" : 0.179,
    "Flexibility": 0.056,
    "Tax_benefits": 0.105,
    "Benefits_death": 0.179,
    "Benefits_survival": 0.149,
    "Good_customer_service": 0.056,
    "Bonus": 0.149,
    "Special_Schemes": 0.021,
    "Riders": 0.105
}

def preference_modeling(user_pref : dict, linguistics: dict) -> dict:
    weights = {}
    preferences = list(user_pref.keys())
    sum_pref = 0
    for pref in preferences:
        importance = user_pref[pref]
        IFNs = linguistic[importance]
        numerator = IFNs[0] + IFNs[2] * (IFNs[0] / (IFNs[0] + IFNs[1]))
        weights[pref] = numerator
        sum_pref += numerator
    for pref in preferences:
        weights[pref] = weights[pref] / sum_pref
    return weights


def generate_grey_relational_coefficient(df, best_policy_value = 100, rho = 0.1):
    d = {}
    policies = df.index.tolist()
    print(policies)
    best_policy_val = 100
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

# corr_df = generate_grey_relational_coefficient(df=df)
# print(corr_df)


def grey_relational_grade(weight, grey_relational_coefficient):
    grade = 0
    policies = grey_relational_coefficient.index.tolist()
    d = {}
    sum_grade = 0
    for policy in policies:
        grade = 0
        for column in df:
            grade += weight[column] * grey_relational_coefficient[column][policy]  
        sum_grade += grade
        d[policy] = grade
        
    for policy in policies:
        d[policy] = d[policy] / sum_grade
        
    return d

def pipline(df, weight, recommendations = 1,  best_policy_value = 100, rho = 0.1):
    corr_df = generate_grey_relational_coefficient(df)
    grade = grey_relational_grade(weight, corr_df)
    sorted_policies = {k: v for k, v in sorted(grade.items(), key=lambda item: item[1])}
    policies = list(sorted_policies.keys())
    return policies[0:recommendations]


weights2 = preference_modeling(user_input, linguistic)
print(pipline(df, weights2))


