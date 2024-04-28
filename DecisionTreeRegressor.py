import os
import pandas as pd
import numpy as np
import joblib as jl
import inspect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tkinter import *
from tkinter import messagebox

os.chdir("C:\\Users\\LENOVO\\Downloads\\IoT_Zadanie")

mergeddata_file = "mergeddata.csv"

if not os.path.exists(mergeddata_file):
    # Прочитане на данните за Павлово
    day_pavlovo_df = pd.read_excel("day.xls",
                                   sheet_name="pavlovo",
                                   header=1,
                                   dtype={"O3": np.float64, "PM10": np.float64}
                                   )

    hour_pavlovo_df = pd.read_excel("hour.xls",
                                    sheet_name="pavlovo",
                                    header=1,
                                    usecols=["Date", "NO", "NO2", "AirTemp", "Press", "UMR"],
                                    dtype={"NO": np.float64, "NO2": np.float64, "AirTemp": np.float64, "Press": np.float64, "UMR": np.float64}
                                    )

    # Обединяване на данните за Павлово
    data_pavlovo_df = pd.merge(hour_pavlovo_df, day_pavlovo_df, on="Date").dropna()

    # Прочитане на данните за Дружба
    day_drujba_df = pd.read_excel("day.xls",
                                  sheet_name="drujba",
                                  header=1,
                                  dtype={"O3": np.float64, "PM10": np.float64}
                                  )

    hour_drujba_df = pd.read_excel("hour.xls",
                                   sheet_name="drujba",
                                   header=1,
                                   usecols=["Date", "NO", "NO2", "AirTemp", "Press", "UMR"],
                                   dtype={"NO": np.float64, "NO2": np.float64, "AirTemp": np.float64, "Press": np.float64, "UMR": np.float64}
                                   )

    # Обединяване на данните за Дружба
    data_drujba_df = pd.merge(hour_drujba_df, day_drujba_df, on="Date").dropna()

    # Прочитане на данните за Хиподрума
    day_hipodruma_df = pd.read_excel("day.xls",
                                     sheet_name="hipodruma",
                                     header=1,
                                     dtype={"O3": np.float64, "PM10": np.float64}
                                     )

    hour_hipodruma_df = pd.read_excel("hour.xls",
                                      sheet_name="hipodruma",
                                      header=1,
                                      usecols=["Date", "NO", "NO2", "AirTemp", "Press", "UMR"],
                                      dtype={"NO": np.float64, "NO2": np.float64, "AirTemp": np.float64, "Press": np.float64, "UMR": np.float64}
                                      )

    # Обединяване на данните за Хиподрума
    data_hipodruma_df = pd.merge(hour_hipodruma_df, day_hipodruma_df, on="Date").dropna()

    # Сливане на всички DataFrame-и в един
    data_df = pd.concat([data_pavlovo_df, data_drujba_df, data_hipodruma_df], ignore_index=True)

    # Закръгляне на числовите колони
    data_df = data_df.round({"NO": 2, "NO2": 2, "O3": 2, "PM10": 2, "AirTemp": 1, "UMR": 1, "Press": 0})

    # Експорт на данните в CSV
    data_df.to_csv(mergeddata_file,
                   columns=["NO", "NO2", "AirTemp", "Press", "UMR", "O3", "PM10"],
                   index=False)

# Дефиниране на пътища за файлове за scaler и model
scaler_file = "scaler.joblib"
model_file = "model.joblib"

# Зареждане на данните
data = pd.read_csv(mergeddata_file)

# Разделяне на свойствата и целевите променливи
x = data[["AirTemp", "Press", "UMR"]]
y = data[["NO", "NO2", "O3", "PM10"]]

# Разделяне на данните на обучение и тест
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Създаване и обучение на трансформатор за скалиране
scaler = RobustScaler()
scaler.fit(x_train)

# Трансформация на данните за обучение и тест
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Запазване на scaler
jl.dump(scaler, scaler_file)

def tune_decision_tree_with_limited_params(x_train_scaled, y_train, x_test_scaled, y_test):
    # Дефиниране на обхватите за стойностите на параметрите
    max_depth_values = [5, 10, 11, 21]
    min_samples_split_values = [2, 10, 28]
    min_samples_leaf_values = [1, 5, 39]
    random_state_values = [0, 42, 100]
    splitter_values = ['best', 'random']

    best_score = float('-inf')  # Инициализация на най-добрия резултат
    best_params = None

    # Итериране през комбинациите на параметрите
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                for random_state in random_state_values:
                    for splitter in splitter_values:
                        # Обучение на модела с текущите стойности на параметрите
                        dt_model = DecisionTreeRegressor(
                            splitter=splitter,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state                            
                        )
                        dt_model.fit(x_train_scaled, y_train)

                        # Оценяване на производителността на модела
                        dt_score = dt_model.score(x_test_scaled, y_test)
                        sel_params = {
                            'splitter': splitter,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': random_state                            
                        }
                        print("R^2 Score:", dt_score, "; Parameters:", sel_params)

                        # Проверка дали текущият модел има по-добър резултат от предишния най-добър
                        if dt_score > best_score:
                            best_score = dt_score
                            best_params = sel_params

    return best_params, best_score

def tune_decision_tree_with_additional_params(x_train_scaled, y_train, x_test_scaled, y_test, base_params):
    # Извличане на базовите параметри
    splitter = base_params['splitter']
    max_depth = base_params['max_depth']
    min_samples_split = base_params['min_samples_split']
    min_samples_leaf = base_params['min_samples_leaf']
    random_state = base_params['random_state']
    
    # Дефиниране на обхватите за стойностите на допълнителните параметри
    criterion_values = ['friedman_mse']
    min_weight_fraction_leaf_values = [0.1, 0.2]
    max_features_values = ['sqrt', 'log2']
    max_leaf_nodes_values = [10, 20]
    min_impurity_decrease_values = [0.1, 0.2]
    ccp_alpha_values = [0.1]

    best_score = float('-inf')  # Инициализация на най-добрия резултат
    best_params = None

    # Итериране през комбинациите на параметрите
    for criterion in criterion_values:
        for min_weight_fraction_leaf in min_weight_fraction_leaf_values:
            for max_features in max_features_values:
                for max_leaf_nodes in max_leaf_nodes_values:
                    for min_impurity_decrease in min_impurity_decrease_values:
                        for ccp_alpha in ccp_alpha_values:
                            # Създаване на текущите параметри
                            current_params = {
                                'criterion': criterion,
                                'splitter': splitter,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                'max_features': max_features,
                                'random_state': random_state,
                                'max_leaf_nodes': max_leaf_nodes,
                                'min_impurity_decrease': min_impurity_decrease,
                                'ccp_alpha': ccp_alpha
                            }

                            # Обучение на модела с текущите стойности на параметрите
                            dt_model = DecisionTreeRegressor(**current_params)
                            dt_model.fit(x_train_scaled, y_train)

                            # Оценяване на производителността на модела
                            dt_score = dt_model.score(x_test_scaled, y_test)
                            print("R^2 Score:", dt_score, "; Parameters:", current_params)

                            # Проверка дали текущият модел има по-добър резултат от предишния най-добър
                            if dt_score > best_score:
                                best_score = dt_score
                                best_params = current_params

    return best_params, best_score

# Получаване на сигнатурата на конструктора на DecisionTreeRegressor
signature = inspect.signature(DecisionTreeRegressor)

# Принтиране на параметрите
for name, param in signature.parameters.items():
    print(f"Parameter: {name}, Default Value: {param.default}")

# Получаване на резултата, когато няма зададени параметри
dt_no_params = DecisionTreeRegressor()
dt_no_params.fit(x_train_scaled, y_train)
dt_no_params_score = dt_no_params.score(x_test_scaled, y_test)
print("\n----------------\n")
print("R^2 Score with No Parameters:", dt_no_params_score)
print("\n----------------\n")

# Получаване на резултата и параметрите с някои зададени:
best_params_limited, best_score_limited = tune_decision_tree_with_limited_params(x_train_scaled, y_train, x_test_scaled, y_test)
print("\n----------------\n")
print("Best Parameters (Limited params):", best_params_limited)
print("Best Score (Limited params):", best_score_limited)
print("\n----------------\n")

# Получаване на резултата и параметрите с всички зададени:
best_params_all, best_score_all = tune_decision_tree_with_additional_params(x_train_scaled, y_train, x_test_scaled, y_test, best_params_limited)
print("\n----------------\n")
print("Best Parameters (All params):", best_params_all)
print("Best Score (All params):", best_score_all)
print("\n----------------\n")

# Получаване на резултата и параметрите с най-високия резултат:
best_score_final, best_params_final = (best_score_all, best_params_all) if best_score_all > best_score_limited else (best_score_limited, best_params_limited)

print("Best Parameters (Final):", best_params_final)
print("Best Score (Final):", best_score_final)
print("\n----------------\n")

# Използване на най-добрите параметри за окончателно обучение на модела
model = DecisionTreeRegressor(**best_params_final)
model.fit(x_train_scaled, y_train)

# Запазване на модела
jl.dump(model, model_file)

final_score = model.score(x_test_scaled, y_test)
print("R^2 Score with Best Parameters:", final_score)

root = Tk()
root.title('Predict Value')
root.resizable(0, 0)

tempL = Label(root, text="Temp(C)")
humL = Label(root, text="Hum(%)")
pressL = Label(root, text="Press(hPa)")
noL = Label(root, text="NO(ug/m3)")
no2L = Label(root, text="NO2(ug/m3)")
ozoneL = Label(root, text="Ozone(ug/m3)")
pm10L = Label(root, text="PM10(ug/m3)")

nonormL = Label(root, text = "NO/NO2 norm is:%d" % 200)
o3normL = Label(root, text = "O3 norm is:%d" % 200)
pm10normL = Label(root, text = "PM10 norm is:%d" % 50)

tempE = Entry(root)
humE = Entry(root)
pressE = Entry(root)
noE = Entry(root)
no2E = Entry(root)
ozoneE = Entry(root)
pm10E = Entry(root)

tempL.grid(row=0, column=0)
tempE.grid(row=0, column=1)

humL.grid(row=1, column=0)
humE.grid(row=1, column=1)

pressL.grid(row=2, column=0)
pressE.grid(row=2, column=1)

noL.grid(row=3, column=0)
noE.grid(row=3, column=1)
nonormL.grid(row=3, column=2)

no2L.grid(row=4, column=0)
no2E.grid(row=4, column=1)

ozoneL.grid(row=5, column=0)
ozoneE.grid(row=5, column=1)
o3normL.grid(row=5, column=2)

pm10L.grid(row=6, column=0)
pm10E.grid(row=6, column=1)
pm10normL.grid(row=6, column=2)

def predict():    
    noE.delete(0,END)
    no2E.delete(0,END)
    ozoneE.delete(0,END)
    pm10E.delete(0,END)
    
    try:
        arr = np.array([[float(tempE.get()),float(humE.get()),float(pressE.get())]])        
        arr_transformed = scaler.transform(arr)        
        pr = model.predict(arr_transformed)

        noE.insert(0,round(pr[0][0],2))
        no2E.insert(0,round(pr[0][1],2))
        ozoneE.insert(0,round(pr[0][2],2))
        pm10E.insert(0,round(pr[0][3],2))        
        
    except ValueError:
        messagebox.showinfo("Wrong Value", "Please enter float values!")

b = Button(root, text="Predict", command=predict)
b.grid(row=7, column=0)

root.mainloop()
