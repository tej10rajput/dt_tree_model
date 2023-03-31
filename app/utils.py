import pickle
import json
import numpy as np
import os 

class Prediction():
    def __init__(self):
        print(os.getcwd())
        

    def load_raw(self):
        with open(r'C:\Users\baps\Desktop\Data Science\Velocity DS 1 Oct\Project\Decision_tree_model\app\wine_model.pkl','rb') as model_file: 
            self.model = pickle.load(model_file)
        
        with open(r'C:\Users\baps\Desktop\Data Science\Velocity DS 1 Oct\Project\Decision_tree_model\app\wine_model.json','r') as col_file: 
            self.column_names = json.load(col_file)
            
        print(f"we are in load raw")

    def predict_quality(self,data):
       
        self.load_raw()
        self.data = data
        user_input = np.zeros(len(self.column_names['Column Names']))
        array = np.array(self.column_names['Column Names'])
        fixed_acidity = self.data['fixed acidity']
        volatile_acidity = self.data['volatile acidity']
        citric_acide = self.data['citric acid']
        residual_sugar = self.data['residual sugar']
        chlorides = self.data['chlorides']
        free_sulfur_dioxide = self.data['free sulfur dioxide']
        total_sulfur_dioxide = self.data['total sulfur dioxide']
        density = self.data['density']
        pH = self.data['pH']
        sulphates = self.data['sulphates']
        alcohol = self.data['alcohol']


        user_input[0] = fixed_acidity
        user_input[1] = volatile_acidity
        user_input[2] = citric_acide
        user_input[3] = residual_sugar
        user_input[4] = chlorides
        user_input[5] = free_sulfur_dioxide
        user_input[6] = total_sulfur_dioxide
        user_input[7] = density 
        user_input[8] = pH 
        user_input[9] = sulphates 
        user_input[10] = alcohol

        print(f"{user_input=}")
        print(len(user_input))

        wine_quality = self.model.predict([user_input])
        print(f"Predicted wine quality = {wine_quality}")

        return wine_quality
    
if __name__ == "__main__":
 
    pred_obj = Prediction()
    pred_obj.load_raw()