import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, te
import lime
import lime.lime_tabular
# import tabular
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


housing = pd.read_csv("cali_housing_data.csv")
housing = housing.drop(columns=["ocean_proximity"])
scaler = StandardScaler()

dropped_data = housing.dropna().reset_index(drop=True)

target = dropped_data["median_house_value"]
temp_data = dropped_data.drop(columns=["median_house_value", "Longitude (East/West)", "Latitude (North/South)", "Population count", "Number of households"])
scaler.fit(temp_data)
normalized_data = scaler.transform(temp_data)
normalized_data = pd.DataFrame(normalized_data, columns=temp_data.columns)
cosine_sim_matrix = cosine_similarity(normalized_data)

# Split the data into features (X) and target (y)
X = temp_data
y = target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the GAM model
# gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + te(0, 1))
gam = LinearGAM(s(0) + s(1) + s(2) + s(3))

# Fit the model to the training data
gam.fit(X_train, y_train)
y_pred = gam.predict(X_test)

# explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X_train.columns, 
#                                                    class_names=['median_house_value'], 
#                                                    verbose=False, mode='regression', discretize_continuous=False)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = X_train.columns, sample_around_instance=True,
                                                   class_names=['median_house_value'], feature_selection='none',
                                                   verbose=True, mode='regression', discretize_continuous=False)

num_instances = X_test.shape[0]  # Get the number of instances in X_test
num_features = X_test.shape[1]

def write_row(samples):
    exp = 0
    ret = []
    for index, row in samples.iterrows():
        instance = [None for _ in range(num_features + 2)]
        explanation = explainer.explain_instance(row.values, gam.predict, num_features=len(X_train.columns))

        x = row
        # if exp == 0:
        #     print(explanation.as_map())
        #     print(explanation.as_list())
            # exp += 1
            # print(x)

        #calculating lime prediction value
        #getting indepedent  variables values
        
        #creating scaled version
        scaled_x = (x - explainer.scaler.mean_) / explainer.scaler.scale_

        #extracting  columns indiced 
        cols=[x[0] for x in explanation.as_map()[1]]
        #extracting their coeff
        coeff=[x[1] for x in explanation.as_map()[1]]
        scaled_coeff = np.array(coeff) * (scaled_x / x).values[cols]
        
        explanation_val = 0
        for feature, weight in zip(cols, scaled_coeff):
            # feature_name = X.columns[feature]

            value = row.values[feature]
            # value = x.iloc[feature]

            # weight = round(weight/1000) * 1000
            # if feature_name == "Average household income":
            #     value *= 1000
            instance[feature] = (value, weight)
            explanation_val += value*weight

        adjustment = round(explanation.intercept[1]/1000) * 1000
        
        explanation_val += adjustment
        print(explanation_val)

        true_value = y_test.loc[index]
        instance[-2], instance[-1] = adjustment, round(true_value/1000) * 1000


        # # Get the feature weights
        # weights = explanation.as_map()[1]  # 1 indicates the positive class in a binary classification problem
        # sorted_weights = sorted(weights, key=lambda x: x[0])

        # for feature, weight in sorted_weights:
        #     feature_name = X.columns[feature]
        #     value = row.values[feature]
        #     weight = round(weight/1000) * 1000
        #     if feature_name == "Average household income":
        #         value *= 1000

        #     instance[feature] = (value, weight)
        #     adjustment = round(explanation.intercept[1]/1000) * 1000
        #     true_value = y_test.loc[index]
        #     instance[-2], instance[-1] = adjustment, round(true_value/1000) * 1000

        ret.append(instance)
    return ret


# d = ["How far west a house is, ranging from -180 (most east) to 180 (most west)",
# "How far north a house is, ranging from -90 (most south) to 90 (most north)",
# "Average age of a house within the block",
# "Total number of rooms within a block",
# "Total number of bedrooms within a block",
# "Total number of people living in a block",
# "Total number of households (home units) in a block",
# "Average income for households within a block of houses (measured in thousands of US Dollars)"]
d = ["Average age of a house within the block",
"Total number of rooms within a block",
"Total number of bedrooms within a block",
"Average income for households within a block of houses (measured in thousands of US Dollars)"]

descriptions = [None for i in range(50)]
descriptions.extend(d)

to_write = [descriptions]
samples = X_test.sample(n=50)

features = write_row(samples)

scaler = StandardScaler()
scaler.fit(samples)
samples = scaler.transform(samples)

# do pca for 4 principal components
num_components = 4

pca = PCA(n_components=num_components)
transformed_data = pca.fit_transform(samples)
cosine_sim = cosine_similarity(transformed_data)

# cosine similarity with pca
for i in range(50):
    similarities = []
    for j in range(50):
        similarities.append(cosine_sim[i, j])
    similarities.extend(features[i])
    to_write.append(similarities)


# cosine similarity without pca

# for i in range(50):
#     current_sample = samples[i]
#     similarities = []
#     for j in range(50):
#         compare_sample = samples[j]
#         similarities.append(cosine_similarity(current_sample.reshape(1,-1), compare_sample.reshape(1,-1))[0][0])
#     similarities.extend(features[i])
#     to_write.append(similarities)

csv_file = "output.csv"

fieldnames = []
for i in range(50):
    fieldnames.append("Instance " + str(i))
for i in range(num_features):
    name = X_test.columns[i]
    fieldnames.append(name)
fieldnames.extend(["adjustment", "true value"])

# Open the file in write mode and create a CSV writer object
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the column names as the first row
    writer.writerow(fieldnames)

    # Write the data rows to the CSV file
    writer.writerows(to_write)