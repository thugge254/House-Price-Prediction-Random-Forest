# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


# Load data
df = pd.read_csv('data/train.csv')


print('The shape of train data :', df.shape)

# Check Columns in the data
for col in df.columns:
    print(col)

# Print a concise structural summary of the dataset.
df.info()

# ======================================================
# Separate Features and Target
# ======================================================

X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]

# ======================================================
# Split the Data (Avoid Data Leakage)
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# # ============================================================
# # Exploratory Data Analysis
# # ============================================================

# # Distribution of SalePrice
# plt.figure(figsize=(8,5))
# sns.histplot(df["SalePrice"], kde=True)
# plt.title("Distribution of SalePrice")
# plt.show()

# # Correlation Heatmap
# plt.figure(figsize=(12,10))
# corr = df.corr(numeric_only=True)
# sns.heatmap(corr)
# plt.title("Correlation Heatmap")
# plt.show()

# # Correlation with target variable
# print("\nCorrelation with SalePrice:")
# print(corr["SalePrice"].sort_values(ascending=False))

# # Scatter Plot
# plt.figure(figsize=(7,5))
# sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
# plt.title("Ground Living Area vs Sale Price")
# plt.show()


# One-Hot Encoding
mssubclass_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Learn the categories and encode the training data
mssubclass_train = mssubclass_ohe.fit_transform(
    X_train[['MSSubClass']]
)

# Create a DataFrame for the encoded columns
mssubclass_df = pd.DataFrame(
    mssubclass_train,
    columns=mssubclass_ohe.get_feature_names_out(['MSSubClass']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['MSSubClass'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, mssubclass_df], axis=1)

# replace all values of 'C (all)' with 'C'
X_train['MSZoning'] = X_train['MSZoning'].replace('C (all)', 'C')

# One-Hot Encoding
mszoning_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

# Learn the categories and encode the training data
mszoning_train = mszoning_ohe.fit_transform(
    X_train[['MSZoning']]
)

# Create a DataFrame for the encoded columns
mszoning_df = pd.DataFrame(
    mszoning_train,
    columns=mszoning_ohe.get_feature_names_out(['MSZoning']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['MSZoning'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, mszoning_df], axis=1)

# Ordinal encoding
street_map = {
    'Grvl': 0,
    'Pave': 1
}

X_train['Street'] = X_train['Street'].map(street_map)

# Replace NA values
X_train["Alley"] = X_train["Alley"].fillna("NoAlley")

# encoding the variable
alley_map = {
    'NA': 0,
    'Grvl': 1,
    'Pave': 2
}

X_train['Alley'] = X_train['Alley'].map(alley_map)

# encoding the variable
lotshape_map = {
    "Reg": 4,
    "IR1": 3,
    "IR2": 2,
    "IR3": 1
}

X_train["LotShape"] = X_train["LotShape"].map(lotshape_map)

# One-Hot Encoding
landcontour_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

landcontour_train = landcontour_ohe.fit_transform(
    X_train[['LandContour']]
)

# Create a DataFrame for the encoded columns
landcontour_df = pd.DataFrame(
    landcontour_train,
    columns=landcontour_ohe.get_feature_names_out(['LandContour']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['LandContour'])


# Concatenate encoded columns
X_train = pd.concat([X_train, landcontour_df], axis=1)

# One-Hot Encoding
utilities_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

utilities_train = utilities_ohe.fit_transform(
    X_train[['Utilities']]
)

# Create a DataFrame for the encoded columns
utilities_df = pd.DataFrame(
    utilities_train,
    columns=utilities_ohe.get_feature_names_out(['Utilities']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Utilities'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, utilities_df], axis=1)

# One-Hot Encoding
lotconfig_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

lotconfig_train = lotconfig_ohe.fit_transform(
    X_train[['LotConfig']]
)

# Create a DataFrame for the encoded columns
lotconfig_df = pd.DataFrame(
    lotconfig_train,
    columns=lotconfig_ohe.get_feature_names_out(['LotConfig']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['LotConfig'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, lotconfig_df], axis=1)

# Ordinal Encoding
landslope_mapping = {
    'Gtl': 1,
    'Mod': 2,
    'Sev': 3
}

X_train['LandSlope'] = X_train['LandSlope'].map(landslope_mapping)

# One-Hot Encoding
neighborhood_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

neighborhood_train = neighborhood_ohe.fit_transform(
    X_train[['Neighborhood']]
)

# Create a DataFrame for the encoded columns
neighborhood_df = pd.DataFrame(
    neighborhood_train,
    columns=neighborhood_ohe.get_feature_names_out(['Neighborhood']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Neighborhood'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, neighborhood_df], axis=1)

# Fill missing values (if any)
X_train["Condition1"] = X_train["Condition1"].fillna("NA")

# One-Hot Encoding
condition1_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
condition1_train = condition1_ohe.fit_transform(
    X_train[["Condition1"]]
)

# Create a DataFrame for the encoded columns
condition1_df = pd.DataFrame(
    condition1_train,
    columns=condition1_ohe.get_feature_names_out(["Condition1"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["Condition1"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, condition1_df], axis=1)

# Fill missing values (if any)
X_train["Condition2"] = X_train["Condition2"].fillna("NA")

# One-Hot Encoding
condition2_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
condition2_train = condition2_ohe.fit_transform(
    X_train[["Condition2"]]
)

# Create a DataFrame for the encoded columns
condition2_df = pd.DataFrame(
    condition2_train,
    columns=condition2_ohe.get_feature_names_out(["Condition2"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["Condition2"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, condition2_df], axis=1)

# One-Hot Encoding
bldgtype_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

bldgtype_train = bldgtype_ohe.fit_transform(
    X_train[['BldgType']]
)

# Create a DataFrame for the encoded columns
bldgtype_df = pd.DataFrame(
    bldgtype_train,
    columns=bldgtype_ohe.get_feature_names_out(['BldgType']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['BldgType'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, bldgtype_df], axis=1)

# One-Hot Encoding
housestyle_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

housestyle_train = housestyle_ohe.fit_transform(
    X_train[['HouseStyle']]
)

# Create a DataFrame for the encoded columns
housestyle_df = pd.DataFrame(
    housestyle_train,
    columns=housestyle_ohe.get_feature_names_out(['HouseStyle']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['HouseStyle'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, housestyle_df], axis=1)

# ordinal encoding through mapping
overallqual_mapping = {
    "Very Poor": 1,
    "Poor": 2,
    "Fair": 3,
    "Below Average": 4,
    "Average": 5,
    "Above Average": 6,
    "Good": 7,
    "Very Good": 8,
    "Excellent": 9,
    "Very Excellent": 10
}

# map the values after encoding
X_train["OverallQual"] = X_train["OverallQual"].map(overallqual_mapping)

# ordinal encoding by mapping
overallcond_mapping = {
    'Very Poor': 1,
    'Poor': 2,
    'Fair': 3,
    'Below Average': 4,
    'Average': 5,
    'Above Average': 6,
    'Good': 7,
    'Very Good': 8,
    'Excellent': 9,
    'Very Excellent': 10
}

# maping the values
X_train['OverallCond'] = X_train['OverallCond'].map(overallcond_mapping)

# One-Hot Encoding
roofstyle_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

roofstyle_train = roofstyle_ohe.fit_transform(
    X_train[['RoofStyle']]
)

# Create a DataFrame for the encoded columns
roofstyle_df = pd.DataFrame(
    roofstyle_train,
    columns=roofstyle_ohe.get_feature_names_out(['RoofStyle']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['RoofStyle'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, roofstyle_df], axis=1)

# One-Hot Encoding
roofmatl_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

roofmatl_train = roofmatl_ohe.fit_transform(
    X_train[['RoofMatl']]
)

# Create a DataFrame for the encoded columns
roofmatl_df = pd.DataFrame(
    roofmatl_train,
    columns=roofmatl_ohe.get_feature_names_out(['RoofMatl']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['RoofMatl'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, roofmatl_df], axis=1)

# One-Hot Encoding
exterior1st_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

exterior1st_train = exterior1st_ohe.fit_transform(
    X_train[['Exterior1st']]
)

# Create a DataFrame for the encoded columns
exterior1st_df = pd.DataFrame(
    exterior1st_train,
    columns=exterior1st_ohe.get_feature_names_out(['Exterior1st']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Exterior1st'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, exterior1st_df], axis=1)

# One-Hot Encoding
exterior2nd_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

exterior2nd_train = exterior2nd_ohe.fit_transform(
    X_train[['Exterior2nd']]
)

# Create a DataFrame for the encoded columns
exterior2nd_df = pd.DataFrame(
    exterior2nd_train,
    columns=exterior2nd_ohe.get_feature_names_out(['Exterior2nd']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Exterior2nd'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, exterior2nd_df], axis=1)

# One-Hot Encoding
masvnr_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False)
masvnr_train = masvnr_ohe.fit_transform(
    X_train[['MasVnrType']]
    )

# create a dataframe for the encoded columns
masvnr_df = pd.DataFrame(
    masvnr_train,
    columns=masvnr_ohe.get_feature_names_out(['MasVnrType']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['MasVnrType'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, masvnr_df], axis=1)


# encoding the variable
exterqual_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

# mapping the values after encoding
X_train['ExterQual'] = X_train['ExterQual'].map(exterqual_mapping)


extercond_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

# mapping the values after encoding
X_train['ExterCond'] = X_train['ExterCond'].map(extercond_mapping)


# One-Hot Encoding to encode the variable
foundation_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

foundation_train = foundation_ohe.fit_transform(
    X_train[['Foundation']]

)

# Convert the encoded array to a DataFrame
foundation_df = pd.DataFrame(
    foundation_train,
    columns=foundation_ohe.get_feature_names_out(['Foundation']),
    index=X_train.index
)

# Drop the original Foundation column
X_train = X_train.drop(columns=['Foundation'])

# Add the encoded columns
X_train = pd.concat([X_train, foundation_df], axis=1)

# Replace missing values with 'N'
X_train['BsmtQual'] = X_train['BsmtQual'].fillna('NA')

bsmtqual_mapping = {
    'NA': 0,  
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['BsmtQual'] = X_train['BsmtQual'].map(bsmtqual_mapping)

# Replace missing values with 'NA'
X_train['BsmtCond'] = X_train['BsmtCond'].fillna('NA')

bsmtcond_mapping = {
    'NA': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['BsmtCond'] = X_train['BsmtCond'].map(bsmtcond_mapping)

# Replace missing values with 'NA' (No Basement)
X_train['BsmtExposure'] = X_train['BsmtExposure'].fillna('NA')

# Ordinal encoding
bsmtexposure_mapping = {
    'NA': 0,   
    'No': 1,  
    'Mn': 2,  
    'Av': 3,  
    'Gd': 4    
}

X_train['BsmtExposure'] = X_train['BsmtExposure'].map(bsmtexposure_mapping)

# Replace missing values with 'NA'
X_train['BsmtFinType1'] = X_train['BsmtFinType1'].fillna('NA')

# Ordinal mapping
bsmtfintype1_mapping = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

X_train['BsmtFinType1'] = X_train['BsmtFinType1'].map(bsmtfintype1_mapping)


# Replace missing values with 'NA'
X_train['BsmtFinType2'] = X_train['BsmtFinType2'].fillna('NA')

bsmtfintype2_mapping = {
    'NA': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}

X_train['BsmtFinType2'] = X_train['BsmtFinType2'].map(bsmtfintype2_mapping)

# One-Hot Encoding
heating_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
heating_train = heating_ohe.fit_transform(
    X_train[["Heating"]]
)

# Create a DataFrame for the encoded columns
heating_df = pd.DataFrame(
    heating_train,
    columns=heating_ohe.get_feature_names_out(["Heating"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["Heating"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, heating_df], axis=1)

heatingqc_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['HeatingQC'] = X_train['HeatingQC'].map(heatingqc_mapping)


centralair_mapping = {
    'N': 0,
    'Y': 1
}

X_train['CentralAir'] = X_train['CentralAir'].map(centralair_mapping)


# Fill missing values first
X_train["Electrical"] = X_train["Electrical"].fillna("NA")

# One-Hot Encoding
electrical_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
electrical_train = electrical_ohe.fit_transform(
    X_train[["Electrical"]]
)

# Create a DataFrame for the encoded columns
electrical_df = pd.DataFrame(
    electrical_train,
    columns=electrical_ohe.get_feature_names_out(["Electrical"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["Electrical"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, electrical_df], axis=1)

kitchenqual_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['KitchenQual'] = X_train['KitchenQual'].map(kitchenqual_mapping)


functional_mapping = {
    'Sal': 1,
    'Sev': 2,
    'Maj2': 3,
    'Maj1': 4,
    'Mod': 5,
    'Min2': 6,
    'Min1': 7,
    'Typ': 8
}

X_train['Functional'] = X_train['Functional'].map(functional_mapping)


# Replace missing values with 'NA'
X_train['FireplaceQu'] = X_train['FireplaceQu'].fillna('NA')

fireplacequ_mapping = {
    'NA': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['FireplaceQu'] = X_train['FireplaceQu'].map(fireplacequ_mapping)


# Fill missing values first
X_train["GarageType"] = X_train["GarageType"].fillna("NA")

# One-Hot Encoding
garagetype_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
garagetype_train = garagetype_ohe.fit_transform(
    X_train[["GarageType"]]
)

# Create a DataFrame for the encoded columns
garagetype_df = pd.DataFrame(
    garagetype_train,
    columns=garagetype_ohe.get_feature_names_out(["GarageType"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["GarageType"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, garagetype_df], axis=1)

# Replace all missing values with 'NA'
X_train['GarageFinish'] = X_train['GarageFinish'].fillna('NA')

# encoding values
garagefinish_mapping = {
    'NA': 0,
    'Unf': 1,
    'RFn': 2,
    'Fin': 3
}

X_train['GarageFinish'] = X_train['GarageFinish'].map(garagefinish_mapping)


# Replace all missing values with 'NA'
X_train['GarageQual'] = X_train['GarageQual'].fillna('NA')

# encoding values
garagequal_mapping = {
    'NA': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

# mapping the values after encoding 
X_train['GarageQual'] = X_train['GarageQual'].map(garagequal_mapping)

# Replace missing values with 'NA'
X_train['GarageCond'] = X_train['GarageCond'].fillna('NA')

garagecond_mapping = {
    'NA': 0,
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

# mapping the values after encoding 
X_train['GarageCond'] = X_train['GarageCond'].map(garagecond_mapping)

# Replace all na values with 'N'
X_train['PavedDrive'] = X_train['PavedDrive'].fillna('N')

paveddrive_mapping = {
    'N': 0,
    'P': 1,
    'Y': 2
}

X_train['PavedDrive'] = X_train['PavedDrive'].map(paveddrive_mapping)

# Replace missing values with 'No Pool'
X_train['PoolQC'] = X_train['PoolQC'].fillna('No Pool')

# encoding the values
poolqc_mapping = {
    'No Pool': 0,
    'Fa': 1,
    'TA': 2,
    'Gd': 3,
    'Ex': 4
}

# map the values 
X_train['PoolQC'] = X_train['PoolQC'].map(poolqc_mapping)

# Replace missing values with 'No Fence'
X_train['Fence'] = X_train['Fence'].fillna('No Fence')

# Encoding the values
fence_mapping = {
    'No Fence': 0,
    'MnWw': 1,
    'GdWo': 2,
    'MnPrv': 3,
    'GdPrv': 4
}

# Map the values after encoding
X_train['Fence'] = X_train['Fence'].map(fence_mapping)

# Fill missing values first
X_train["MiscFeature"] = X_train["MiscFeature"].fillna("None")

# One-Hot Encoding
miscfeature_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
miscfeature_train = miscfeature_ohe.fit_transform(
    X_train[["MiscFeature"]]
)

# Create a DataFrame for the encoded columns
miscfeature_df = pd.DataFrame(
    miscfeature_train,
    columns=miscfeature_ohe.get_feature_names_out(["MiscFeature"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["MiscFeature"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, miscfeature_df], axis=1)

# One-Hot Encoding
saletype_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
saletype_train = saletype_ohe.fit_transform(
    X_train[["SaleType"]]
)

# Create a DataFrame for the encoded columns
saletype_df = pd.DataFrame(
    saletype_train,
    columns=saletype_ohe.get_feature_names_out(["SaleType"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["SaleType"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, saletype_df], axis=1)

# One-Hot Encoding
salecondition_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

# Learn the categories and encode the training data
salecondition_train = salecondition_ohe.fit_transform(
    X_train[["SaleCondition"]]
)

# Create a DataFrame for the encoded columns
salecondition_df = pd.DataFrame(
    salecondition_train,
    columns=salecondition_ohe.get_feature_names_out(["SaleCondition"]),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=["SaleCondition"])

# Concatenate the encoded columns
X_train = pd.concat([X_train, salecondition_df], axis=1)

################ Handling missing values ######################

X_train["GarageYrBlt"] = X_train["GarageYrBlt"].fillna(0)

# compute median value for LotFrontage
lotfrontage_median = X_train["LotFrontage"].median()

# replace all na values with median score
X_train["LotFrontage"] = X_train["LotFrontage"].fillna(lotfrontage_median)

X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)











# Transform the test data
mssubclass_test = mssubclass_ohe.transform(
    X_test[['MSSubClass']]
)

# Create a DataFrame for the encoded columns
mssubclass_test_df = pd.DataFrame(
    mssubclass_test,
    columns=mssubclass_ohe.get_feature_names_out(['MSSubClass']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['MSSubClass'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, mssubclass_test_df], axis=1)

# replace all values of 'C (all)' with 'C'
X_test['MSZoning'] = X_test['MSZoning'].replace('C (all)', 'C')

# Encode the test data using the encoder fitted on the training data
mszoning_test = mszoning_ohe.transform(
    X_test[['MSZoning']]
)

# Create a DataFrame for the encoded columns
mszoning_test_df = pd.DataFrame(
    mszoning_test,
    columns=mszoning_ohe.get_feature_names_out(['MSZoning']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['MSZoning'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, mszoning_test_df], axis=1)

# map the data after encoding
X_test['Alley'] = X_test['Alley'].map(alley_map)

# map the data after encoding
X_test['Street'] = X_test['Street'].map(street_map)


# map the data after encoding
X_test["LotShape"] = X_test["LotShape"].map(lotshape_map)

# Transform the test data
landcontour_test = landcontour_ohe.transform(
    X_test[['LandContour']]
)

# Create a DataFrame for the encoded columns
landcontour_test_df = pd.DataFrame(
    landcontour_test,
    columns=landcontour_ohe.get_feature_names_out(['LandContour']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['LandContour'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, landcontour_test_df], axis=1)

# Transform the test data
utilities_test = utilities_ohe.transform(
    X_test[['Utilities']]
)

# Create a DataFrame for the encoded columns
utilities_test_df = pd.DataFrame(
    utilities_test,
    columns=utilities_ohe.get_feature_names_out(['Utilities']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['Utilities'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, utilities_test_df], axis=1)

# Transform the test data
lotconfig_test = lotconfig_ohe.transform(
    X_test[['LotConfig']]
)

# Create a DataFrame for the encoded columns
lotconfig_test_df = pd.DataFrame(
    lotconfig_test,
    columns=lotconfig_ohe.get_feature_names_out(['LotConfig']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['LotConfig'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, lotconfig_test_df], axis=1)

# map the values after encoding
X_test['LandSlope'] = X_test['LandSlope'].map(landslope_mapping)

# Transform the test data
neighborhood_test = neighborhood_ohe.transform(
    X_test[['Neighborhood']]
)

# Create a DataFrame for the encoded columns
neighborhood_test_df = pd.DataFrame(
    neighborhood_test,
    columns=neighborhood_ohe.get_feature_names_out(['Neighborhood']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['Neighborhood'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, neighborhood_test_df], axis=1)

# Fill missing values (if any)
X_test["Condition1"] = X_test["Condition1"].fillna("NA")

# Encode the test data using the encoder fitted on the training data
condition1_test = condition1_ohe.transform(
    X_test[["Condition1"]]
)

# Create a DataFrame for the encoded columns
condition1_test_df = pd.DataFrame(
    condition1_test,
    columns=condition1_ohe.get_feature_names_out(["Condition1"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["Condition1"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, condition1_test_df], axis=1)

# Fill missing values (if any)
X_test["Condition2"] = X_test["Condition2"].fillna("NA")

# Encode the test data using the encoder fitted on the training data
condition2_test = condition2_ohe.transform(
    X_test[["Condition2"]]
)

# Create a DataFrame for the encoded columns
condition2_test_df = pd.DataFrame(
    condition2_test,
    columns=condition2_ohe.get_feature_names_out(["Condition2"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["Condition2"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, condition2_test_df], axis=1)

# Transform the test data
bldgtype_test = bldgtype_ohe.transform(
    X_test[['BldgType']]
)

# Create a DataFrame for the encoded columns
bldgtype_test_df = pd.DataFrame(
    bldgtype_test,
    columns=bldgtype_ohe.get_feature_names_out(['BldgType']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['BldgType'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, bldgtype_test_df], axis=1)



# Transform the test data
housestyle_test = housestyle_ohe.transform(
    X_test[['HouseStyle']]
)

# Create a DataFrame for the encoded columns
housestyle_test_df = pd.DataFrame(
    housestyle_test,
    columns=housestyle_ohe.get_feature_names_out(['HouseStyle']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['HouseStyle'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, housestyle_test_df], axis=1)

# map the variables using the encoding function created on training phase
X_test["OverallQual"] = X_test["OverallQual"].map(overallqual_mapping)

# map the variables using the encoding function created on training phase
X_test['OverallCond'] = X_test['OverallCond'].map(overallcond_mapping)


# Transform the test data
roofstyle_test = roofstyle_ohe.transform(
    X_test[['RoofStyle']]
)

# Create a DataFrame for the encoded columns
roofstyle_test_df = pd.DataFrame(
    roofstyle_test,
    columns=roofstyle_ohe.get_feature_names_out(['RoofStyle']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['RoofStyle'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, roofstyle_test_df], axis=1)

# Transform the test data
roofmatl_test = roofmatl_ohe.transform(
    X_test[['RoofMatl']]
)

# Create a DataFrame for the encoded columns
roofmatl_test_df = pd.DataFrame(
    roofmatl_test,
    columns=roofmatl_ohe.get_feature_names_out(['RoofMatl']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['RoofMatl'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, roofmatl_test_df], axis=1)

# Transform the test data
exterior1st_test = exterior1st_ohe.transform(
    X_test[['Exterior1st']]
)

# Create a DataFrame for the encoded columns
exterior1st_test_df = pd.DataFrame(
    exterior1st_test,
    columns=exterior1st_ohe.get_feature_names_out(['Exterior1st']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['Exterior1st'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, exterior1st_test_df], axis=1)

# Transform the test data
exterior2nd_test = exterior2nd_ohe.transform(
    X_test[['Exterior2nd']]
)

# Create a DataFrame for the encoded columns
exterior2nd_test_df = pd.DataFrame(
    exterior2nd_test,
    columns=exterior2nd_ohe.get_feature_names_out(['Exterior2nd']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['Exterior2nd'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, exterior2nd_test_df], axis=1)

# Replace missing values with "None"
X_test["MasVnrType"] = X_test["MasVnrType"].fillna("None")

# Transform the test data
masvnr_test = masvnr_ohe.transform(
    X_test[["MasVnrType"]]
)

# Create a DataFrame
masvnr_test_df = pd.DataFrame(
    masvnr_test,
    columns=masvnr_ohe.get_feature_names_out(["MasVnrType"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["MasVnrType"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, masvnr_test_df], axis=1)


# map the values 
X_test["ExterQual"] = X_test["ExterQual"].map(exterqual_mapping)

# map the values after encoding 
X_test['ExterCond'] = X_test['ExterCond'].map(extercond_mapping)


# Transform the test data
foundation_test = foundation_ohe.transform(
    X_test[["Foundation"]]
)

# Create a DataFrame
foundation_test_df = pd.DataFrame(
    foundation_test,
    columns=foundation_ohe.get_feature_names_out(["Foundation"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["Foundation"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, foundation_test_df], axis=1)



# Replace missing values with 'N'
X_test['BsmtQual'] = X_test['BsmtQual'].fillna('NA')

# map the values after encoding 
X_test['BsmtQual'] = X_test['BsmtQual'].map(bsmtqual_mapping)

# map the values after encoding
X_test['BsmtCond'] = X_test['BsmtCond'].map(bsmtcond_mapping)

# Replace missing values with 'NA' (No Basement)
X_test['BsmtExposure'] = X_test['BsmtExposure'].fillna('NA')

# map the values after encoding 
X_test['BsmtExposure'] = X_test['BsmtExposure'].map(bsmtexposure_mapping)

# Replace missing values with 'NA'
X_test['BsmtFinType1'] = X_test['BsmtFinType1'].fillna('NA')

# map the values after encoding and convert to categorical
X_test['BsmtFinType1'] = X_test['BsmtFinType1'].map(bsmtfintype1_mapping)

# map the values after encoding and convert to categorical
X_test['BsmtFinType2'] = X_test['BsmtFinType2'].map(bsmtfintype2_mapping)



# Encode the test data using the encoder fitted on the training data
heating_test = heating_ohe.transform(
    X_test[["Heating"]]
)

# Create a DataFrame for the encoded columns
heating_test_df = pd.DataFrame(
    heating_test,
    columns=heating_ohe.get_feature_names_out(["Heating"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["Heating"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, heating_test_df], axis=1)

# map the values after encoding and convert to categorical
X_test['HeatingQC'] = X_test['HeatingQC'].map(heatingqc_mapping)

# map the values after encoding 
X_test['CentralAir'] = X_test['CentralAir'].map(centralair_mapping)

# Fill missing values first
X_test["Electrical"] = X_test["Electrical"].fillna("NA")

# Encode the test data using the encoder fitted on the training data
electrical_test = electrical_ohe.transform(
    X_test[["Electrical"]]
)

# Create a DataFrame for the encoded columns
electrical_test_df = pd.DataFrame(
    electrical_test,
    columns=electrical_ohe.get_feature_names_out(["Electrical"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["Electrical"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, electrical_test_df], axis=1)

# map the values after encoding
X_test['KitchenQual'] = X_test['KitchenQual'].map(kitchenqual_mapping)

# Mapping the values after encoding
X_test['Functional'] = X_test['Functional'].map(functional_mapping)

# rmaping the data values after encoding
X_test['FireplaceQu'] = X_test['FireplaceQu'].map(fireplacequ_mapping)


################### Handling missing values #########################################
X_test["GarageYrBlt"] = X_test["GarageYrBlt"].fillna(0)

# compute median value for LotFrontage
lotfrontage_median = X_test["LotFrontage"].median()

# replace all na values with median score
X_test["LotFrontage"] = X_test["LotFrontage"].fillna(lotfrontage_median)

# Fill missing values first
X_test["GarageType"] = X_test["GarageType"].fillna("NA")

# Encode the test data using the encoder fitted on the training data
garagetype_test = garagetype_ohe.transform(
    X_test[["GarageType"]]
)

# Create a DataFrame for the encoded columns
garagetype_test_df = pd.DataFrame(
    garagetype_test,
    columns=garagetype_ohe.get_feature_names_out(["GarageType"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["GarageType"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, garagetype_test_df], axis=1)

# Replace all missing values with 'NA'
X_test['GarageFinish'] = X_test['GarageFinish'].fillna('NA')

# mapping the values after encoding
X_test['GarageFinish'] = X_test['GarageFinish'].map(garagefinish_mapping)


# Replace all missing values with 'NA'
X_test['GarageQual'] = X_test['GarageQual'].fillna('NA')

# map the data values after encoding 
X_test['GarageQual'] = X_test['GarageQual'].map(garagequal_mapping)

# map the da values after encoding
X_test['GarageCond'] = X_test['GarageCond'].map(garagecond_mapping)

# Replace na values with 'N'
X_test['PavedDrive'] = X_test['PavedDrive'].fillna('N')

# map the data values after encoding 
X_test['PavedDrive'] = X_test['PavedDrive'].map(paveddrive_mapping)

# Replace missing values with 'No Pool'
X_test['PoolQC'] = X_test['PoolQC'].fillna('No Pool')

# map the values 
X_test['PoolQC'] = X_test['PoolQC'].map(poolqc_mapping)

# Replace missing values with 'No Fence'
X_test['Fence'] = X_test['Fence'].fillna('No Fence')

# Map the values after encoding 
X_test['Fence'] = X_test['Fence'].map(fence_mapping)

# Fill missing values first
X_test["MiscFeature"] = X_test["MiscFeature"].fillna("None")

# Encode the test data using the encoder fitted on the training data
miscfeature_test = miscfeature_ohe.transform(
    X_test[["MiscFeature"]]
)

# Create a DataFrame for the encoded columns
miscfeature_test_df = pd.DataFrame(
    miscfeature_test,
    columns=miscfeature_ohe.get_feature_names_out(["MiscFeature"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["MiscFeature"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, miscfeature_test_df], axis=1)

# Encode the test data using the fitted encoder
saletype_test = saletype_ohe.transform(
    X_test[["SaleType"]]
)

# Create a DataFrame for the encoded columns
saletype_test_df = pd.DataFrame(
    saletype_test,
    columns=saletype_ohe.get_feature_names_out(["SaleType"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["SaleType"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, saletype_test_df], axis=1)

# Encode the test data using the encoder fitted on the training data
salecondition_test = salecondition_ohe.transform(
    X_test[["SaleCondition"]]
)

# Create a DataFrame for the encoded columns
salecondition_test_df = pd.DataFrame(
    salecondition_test,
    columns=salecondition_ohe.get_feature_names_out(["SaleCondition"]),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=["SaleCondition"])

# Concatenate the encoded columns
X_test = pd.concat([X_test, salecondition_test_df], axis=1)


print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")




###############################################
######## Regression Random Forest Model########
###############################################

# print the shape of the data
print("The shape of training data is :", X_train.shape)
print("The shape of testing data is :", X_test.shape)

# create rf model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set data
y_pred = rf_model.predict(X_test)

print(f"The Shape of the prected values: {y_pred.shape}")



###############################################
############# evaluation metrics ##############
###############################################
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")



###############################################
#### compare actual vs. predicted values ######
###############################################
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

comparison["Error"] = comparison["Actual Price"] - comparison["Predicted Price"]
comparison["Absolute Error"] = comparison["Error"].abs()

print(comparison.head(10))



feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})


###############################################
############## Feature Importance #############
###############################################
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(20))


###############################################
############ Plot Feature Importance ##########
###############################################
custom_colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e74c3c"]
top_features = feature_importance.head(15)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"], top_features["Importance"], color=custom_colors)
plt.xlabel("Importance")
plt.title("Top 15 Most Important Features")
plt.gca().invert_yaxis()
plt.show()


###############################################
############ Predicted vs Actual Plot #########
###############################################
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


###############################################
############### Residual Plot #################
###############################################
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()
