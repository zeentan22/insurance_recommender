from recommender import InsuranceRecommender
import pandas as pd


linguistic = {
    "Very Important" : [0.9,0.05,0.05],
    "Important" : [0.75, 0.2, 0.05],
    "Medium" : [0.5, 0.4, 0.1],
    "Unimportant" : [0.25, 0.6, 0.15],
    "Very unimportant": [0.1, 0.8, 0.1]
}


d = {"Low_Premium": [100,80,80,72,72,92,86], 
     "Flexibility": [0,100,100,0,0,75,50],
      "Tax_benefits": [60,80,80,86,86,64,70],
      "Benefits_death": [66,85,85,86,86,70,74],
      "Benefits_survival": [64,90,90,88,88,74,80],
      "Good_customer_service": [80,80,80,80,80,80,80],
      "Bonus": [80,100,100,90,90,90,100],
      "Special_Schemes": [60,60,60,80,80,0,0],
      "Riders":
      [100,0,100,0,100,100,100],
      "Entry Age":
      [21,21,25,21,25, 21, 21]}

df = pd.DataFrame(data = d, index = ["P1",'P2',"P3",'P4',"P5", 'P6','P7'])
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

user_demographics = {
    "Age":23,
}

full_input = [user_demographics, user_input]

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


if __name__ == "__main__":
    recommender = InsuranceRecommender(linguistic, full_input , df)
    recommender.df = df
    print(recommender.run(1))

