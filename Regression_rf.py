# import packages
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


# Load data
X_train = pd.read_csv('data/train.csv')


print('The shape of train data :', X_train.shape)


# Check Columns in the data
print(X_train.columns)

for col in X_train.columns:
    print(col)

# Print a concise structural summary of the dataset.
X_train.info()

# show number of rows and columns in the data
print("Number of rows:", X_train.shape[0])
print("Number of columns:", X_train.shape[1])

# convert MSSubClass to categorical variable
X_train["MSSubClass"] = X_train["MSSubClass"].astype("category")

# replace all values of 'C (all)' with 'C'
X_train['MSZoning'] = X_train['MSZoning'].replace('C (all)', 'C')

# One-Hot Encoding.
mszoning_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

mszoning_train = mszoning_ohe.fit_transform(
    X_train[['MSZoning']]
)

# create a dataframe for the encoded columns
mszoning_df = pd.DataFrame(
    mszoning_train,
    columns=mszoning_ohe.get_feature_names_out(['MSZoning']),
    index=X_train.index
)

# Drop original column
X_train = X_train.drop(columns=['MSZoning'])

# concatenate the encoded columns to X_train data
X_train = pd.concat([X_train, mszoning_df], axis=1)

# Ordinal encoding
street_map = {
    'Grvl': 0,
    'Pave': 1
}

X_train['Street'] = X_train['Street'].map(street_map)

# convert Street to categorical variable
X_train["Street"] = X_train["Street"].astype("category")

# Replace NA values
X_train["Alley"] = X_train["Alley"].fillna("NoAlley")

# encoding the variable
alley_map = {
    'NA': 0,
    'Grvl': 1,
    'Pave': 2
}

X_train['Alley'] = X_train['Alley'].map(alley_map)

# convert Alley to categorical variable
X_train["Alley"] = X_train["Alley"].astype("category")

# Replace NA values
X_train['MiscFeature'] = X_train['MiscFeature'].fillna('None')

# convert MiscFeature to categorical variable
X_train["MiscFeature"] = X_train["MiscFeature"].astype("category")


# convert LotShape to categorical variable
X_train["LotShape"] = X_train["LotShape"].astype("category")

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

# One-Hot Encoding
condition1_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

condition1_train = condition1_ohe.fit_transform(
    X_train[['Condition1']]
)

# Create a DataFrame for the encoded columns
condition1_df = pd.DataFrame(
    condition1_train,
    columns=condition1_ohe.get_feature_names_out(['Condition1']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Condition1'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, condition1_df], axis=1)

# One-Hot Encoding
condition2_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

condition2_train = condition2_ohe.fit_transform(
    X_train[['Condition2']]
)

# Create a DataFrame for the encoded columns
condition2_df = pd.DataFrame(
    condition2_train,
    columns=condition2_ohe.get_feature_names_out(['Condition2']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['Condition2'])

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

# convert OverallQual to categorical variable
X_train["OverallQual"] = X_train["OverallQual"].astype("category")

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

# convert RoofStyle to categorical variable
X_train["RoofStyle"] = X_train["RoofStyle"].astype("category")

roof_mapping = {
    'CompShg': 'Composite_Shingle',
    'Tar&Grv': 'Tar_Gravel',
    'WdShngl': 'Wood',
    'WdShake': 'Wood',
    'ClyTile': 'Other',
    'Membran': 'Other',
    'Metal': 'Other',
    'Roll': 'Other'
}

X_train['RoofMatl'] = X_train['RoofMatl'].map(roof_mapping)

# convert RoofMatl to categorical variable
X_train["RoofMatl"] = X_train["RoofMatl"].astype("category")

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

# convert ExterQual to categorical variable
X_train["ExterQual"] = X_train["ExterQual"].astype("category")

# convert ExterQual to categorical variable
X_train["ExterQual"] = X_train["ExterQual"].astype("category")

# encoding the variable
exterqual_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

# convert ExterCond to categorical variable
X_train['ExterCond'] = X_train['ExterCond'].astype("category")

extercond_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['ExterCond'] = X_train['ExterCond'].map(extercond_mapping)

# convert Foundation to categorical variable
X_train["Foundation"] = X_train["Foundation"].astype("category")

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

# convert BsmtQual to categorical variable
X_train["BsmtQual"] = X_train["BsmtQual"].astype("category")

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

# convert BsmtCond to categorical variable
X_train["BsmtCond"] = X_train["BsmtCond"].astype("category")

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

# convert BsmtExposure to categorical variable
X_train["BsmtExposure"] = X_train["BsmtExposure"].astype("category")

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

# convert BsmtFinType1 to categorical variable
X_train["BsmtFinType1"] = X_train["BsmtFinType1"].astype("category")

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

# convert BsmtFinType2 to categorical variable
X_train["BsmtFinType2"] = X_train["BsmtFinType2"].astype("category")

# Convert to categorical variable
X_train["Heating"] = X_train["Heating"].astype("category")

# One-Hot Encode
heating_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

heating_train = heating_ohe.fit_transform(
    X_train[['Heating']]
)

# Convert the encoded array to a DataFrame:
heating_df = pd.DataFrame(
    heating_train,
    columns=heating_ohe.get_feature_names_out(['Heating']),
    index=X_train.index
)

# Drop the original column:
X_train = X_train.drop(columns=['Heating'])

# Add the encoded columns to X_train data
X_train = pd.concat([X_train, heating_df], axis=1)

heatingqc_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['HeatingQC'] = X_train['HeatingQC'].map(heatingqc_mapping)

# Convert to categorical variable
X_train["HeatingQC"] = X_train["HeatingQC"].astype("category")

centralair_mapping = {
    'N': 0,
    'Y': 1
}

X_train['CentralAir'] = X_train['CentralAir'].map(centralair_mapping)

# Convert to categorical variable
X_train["CentralAir"] = X_train["CentralAir"].astype("category")


# One-Hot Encoding
electrical_ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

electrical_train = electrical_ohe.fit_transform(
    X_train[['Electrical']]
)

# Convert to a DataFrame:
electrical_df = pd.DataFrame(
    electrical_train,
    columns=electrical_ohe.get_feature_names_out(['Electrical']),
    index=X_train.index
)

# Drop the original column:
X_train = X_train.drop(columns=['Electrical'])

# Add the encoded columns to X_train data
X_train = pd.concat([X_train, electrical_df], axis=1)

kitchenqual_mapping = {
    'Po': 1,
    'Fa': 2,
    'TA': 3,
    'Gd': 4,
    'Ex': 5
}

X_train['KitchenQual'] = X_train['KitchenQual'].map(kitchenqual_mapping)

# Convert to categorical variable
X_train['KitchenQual'] = X_train['KitchenQual'].astype('category')

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

# Convert to categorical variable
X_train['Functional'] = X_train['Functional'].astype('category')

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

# Convert to categorical variable
X_train['FireplaceQu'] = X_train['FireplaceQu'].astype('category')

# One-Hot Encoding
garagetype_ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

garagetype_train = garagetype_ohe.fit_transform(
    X_train[['GarageType']]
)

# Replace missing values with 'NA' and convert to categorical variable
X_train['GarageType'] = X_train['GarageType'].fillna('NA')

# Convert the encoded columns to a DataFrame:
garagetype_train_df = pd.DataFrame(
    garagetype_train,
    columns=garagetype_ohe.get_feature_names_out(['GarageType']),
    index=X_train.index
)

# Drop the original column:
X_train = X_train.drop(columns=['GarageType'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, garagetype_train_df], axis=1)

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

# Convert to categorical variable
X_train['GarageFinish'] = X_train['GarageFinish'].astype('category')

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

X_train['GarageQual'] = X_train['GarageQual'].map(garagequal_mapping).astype('category')

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

X_train['GarageCond'] = X_train['GarageCond'].map(garagecond_mapping).astype('category')

X_train['PavedDrive'] = X_train['PavedDrive'].fillna('N')

paveddrive_mapping = {
    'N': 0,
    'P': 1,
    'Y': 2
}

X_train['PavedDrive'] = X_train['PavedDrive'].map(paveddrive_mapping).astype('category')

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

# map the values and convert to categorical variable
X_train['PoolQC'] = X_train['PoolQC'].map(poolqc_mapping).astype('category')

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

# Map the values and convert to categorical
X_train['Fence'] = X_train['Fence'].map(fence_mapping).astype('category')

# One-Hot Encoding
miscfeature_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

miscfeature_train = miscfeature_ohe.fit_transform(
    X_train[['MiscFeature']].fillna('No Pool')
)

# create a dataframe for the encoded columns
miscfeature_df = pd.DataFrame(
    miscfeature_train,
    columns=miscfeature_ohe.get_feature_names_out(['MiscFeature']),
    index=X_train.index
)

# Drop original column
X_train = X_train.drop(columns=['MiscFeature'])

# Concatenate encoded columns
X_train = pd.concat([X_train, miscfeature_df], axis=1)

# One-Hot Encoding
saletype_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

saletype_train = saletype_ohe.fit_transform(
    X_train[['SaleType']]
)

# create a dataframe for the encoded columns
saletype_df = pd.DataFrame(
    saletype_train,
    columns=saletype_ohe.get_feature_names_out(['SaleType']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['SaleType'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, saletype_df], axis=1)

# One-Hot Encoding
salecondition_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

salecondition_train = salecondition_ohe.fit_transform(
    X_train[['SaleCondition']]
)

# create a dataframe for the encoded columns
salecondition_df = pd.DataFrame(
    salecondition_train,
    columns=salecondition_ohe.get_feature_names_out(['SaleCondition']),
    index=X_train.index
)

# Drop the original column
X_train = X_train.drop(columns=['SaleCondition'])

# Concatenate the encoded columns
X_train = pd.concat([X_train, salecondition_df], axis=1)

# Handling missing values

X_train["GarageYrBlt"] = X_train["GarageYrBlt"].fillna(0)

# compute median value for LotFrontage
lotfrontage_median = X_train["LotFrontage"].median()

# replace all na values with median score
X_train["LotFrontage"] = X_train["LotFrontage"].fillna(lotfrontage_median)

X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)











# Load data
X_test = pd.read_csv('data/test.csv')

print('The shape of test data :', X_test.shape)

# convert MSSubClass to categorical variable
X_test["MSSubClass"] = X_test["MSSubClass"].astype("category")

# replace all values of 'C (all)' with 'C'
X_test['MSZoning'] = X_test['MSZoning'].replace('C (all)', 'C')

# One-Hot Encoding.
mszoning_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

mszoning_test = mszoning_ohe.fit_transform(
    X_test[['MSZoning']]
)

# create a dataframe for the encoded columns
mszoning_test_df = pd.DataFrame(
    mszoning_test,
    columns=mszoning_ohe.get_feature_names_out(['MSZoning']),
    index=X_test.index
)
# Drop original column
X_test = X_test.drop(columns=['MSZoning'])

# Add the encoded columns
X_test = pd.concat([X_test, mszoning_test_df], axis=1)

# map the data after encoding
X_test['Alley'] = X_test['Alley'].map(alley_map)

# convert Alley to categorical variable
X_test["Alley"] = X_test["Alley"].astype("category")

# map the data after encoding
X_test['Street'] = X_test['Street'].map(street_map)

# convert Street to categorical variable
X_test["Street"] = X_test["Street"].astype("category")

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

# Concatenate encoded columns
X_test = pd.concat([X_test, landcontour_test_df], axis=1)

# One-Hot Encoding
garagetype_ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

garagetype_test = garagetype_ohe.fit_transform(
    X_test[['GarageType']]
)

# Replace missing values with 'NA' and convert to categorical variable
X_test['GarageType'] = X_test['GarageType'].fillna('NA')

# Convert the encoded columns to a DataFrame:
garagetype_train_df = pd.DataFrame(
    garagetype_test,
    columns=garagetype_ohe.get_feature_names_out(['GarageType']),
    index=X_test.index
)

# Drop the original column:
X_test = X_test.drop(columns=['GarageType'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, garagetype_train_df], axis=1)

# Replace all missing values with 'NA'
X_test['GarageFinish'] = X_test['GarageFinish'].fillna('NA')

# mapping the values after encoding
X_test['GarageFinish'] = X_test['GarageFinish'].map(garagefinish_mapping)

# Convert to categorical variable
X_test['GarageFinish'] = X_test['GarageFinish'].astype('category')

# Replace all missing values with 'NA'
X_test['GarageQual'] = X_test['GarageQual'].fillna('NA')

# map the data values after encoding and conver to categorical variable
X_test['GarageQual'] = X_test['GarageQual'].map(garagequal_mapping).astype('category')

# Replace na values with 'N'
X_test['PavedDrive'] = X_test['PavedDrive'].fillna('N')

# map the data values after encoding and convert to categorical
X_test['PavedDrive'] = X_test['PavedDrive'].map(paveddrive_mapping).astype('category')

# Replace missing values with 'No Pool'
X_test['PoolQC'] = X_test['PoolQC'].fillna('No Pool')

# map the values and convert to categorical variable
X_test['PoolQC'] = X_test['PoolQC'].map(poolqc_mapping).astype('category')

# Replace missing values with 'No Fence'
X_test['Fence'] = X_test['Fence'].fillna('No Fence')

# Map the values and convert to categorical
X_test['Fence'] = X_test['Fence'].map(fence_mapping).astype('category')

# One-Hot Encoding
miscfeature_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

miscfeature_test = miscfeature_ohe.fit_transform(
    X_test[['MiscFeature']].fillna('No Pool')
)

# create a dataframe for the encoded columns
miscfeature_test_df = pd.DataFrame(
    miscfeature_test,
    columns=miscfeature_ohe.get_feature_names_out(['MiscFeature']),
    index=X_test.index
)

# Drop original column
X_test = X_test.drop(columns=['MiscFeature'])

# Concatenate encoded columns
X_test = pd.concat([X_test, miscfeature_test_df], axis=1)

# One-Hot Encoding
saletype_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

saletype_test = saletype_ohe.fit_transform(
    X_test[['SaleType']]
)

# create a dataframe for the encoded columns
saletype_test_df = pd.DataFrame(
    saletype_test,
    columns=saletype_ohe.get_feature_names_out(['SaleType']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['SaleType'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, saletype_test_df], axis=1)

# One-Hot Encoding
salecondition_ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

salecondition_test = salecondition_ohe.fit_transform(
    X_test[['SaleCondition']]
)

# create a dataframe for the encoded columns
salecondition_test_df = pd.DataFrame(
    salecondition_test,
    columns=salecondition_ohe.get_feature_names_out(['SaleCondition']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['SaleCondition'])

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

# Transform the test data
condition1_test = condition1_ohe.transform(
    X_test[['Condition1']]
)

# Create a DataFrame for the encoded columns
condition1_test_df = pd.DataFrame(
    condition1_test,
    columns=condition1_ohe.get_feature_names_out(['Condition1']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['Condition1'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, salecondition_test_df], axis=1)

# Transform the test data
condition2_test = condition2_ohe.transform(
    X_test[['Condition2']]
)

# create a dataframe for the encoded columns
condition2_test_df = pd.DataFrame(
    condition2_test,
    columns=condition2_ohe.get_feature_names_out(['Condition2']),
    index=X_test.index
)

# drop the original column
X_test = X_test.drop(columns=['Condition2'])

# merge the encoded columns to the original data
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

# map the values after encoding
X_test['LandSlope'] = X_test['LandSlope'].map(landslope_mapping)

# Map the values and convert to categorical variable
X_test['RoofMatl'] = X_test['RoofMatl'].map(roof_mapping).astype('category')

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

# One-Hot Encoding
masvnr_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
    )

masvnr_test = masvnr_ohe.fit_transform(
    X_test[['MasVnrType']]
    )

# create a dataframe for the encoded columns
masvnr_test_df = pd.DataFrame(
    masvnr_test,
    columns=masvnr_ohe.get_feature_names_out(['MasVnrType']),
    index=X_test.index
)

# Drop the original column
X_test = X_test.drop(columns=['MasVnrType'])

# Concatenate the encoded columns
X_test = pd.concat([X_test, masvnr_test_df], axis=1)

# map the values and convert "ExterQual" to categorical variable
X_test["ExterQual"] = X_test["ExterQual"].map(exterqual_mapping).astype("category")

# map the values after encoding and convert to categorical
X_test['ExterCond'] = X_test['ExterCond'].map(extercond_mapping).astype('category')

# One-Hot Encoding to encode the variable
foundation_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

foundation_test = foundation_ohe.fit_transform(
    X_test[['Foundation']]

)

# Convert the encoded array to a DataFrame
foundation_test_df = pd.DataFrame(
    foundation_test,
    columns=foundation_ohe.get_feature_names_out(['Foundation']),
    index=X_test.index
)

# Drop the original Foundation column
X_test = X_test.drop(columns=['Foundation'])

# Add the encoded columns
X_test = pd.concat([X_test, foundation_test_df], axis=1)

# Replace missing values with 'N'
X_test['BsmtQual'] = X_test['BsmtQual'].fillna('NA')

# map the values after encoding and convert to categorical
X_test['BsmtQual'] = X_test['BsmtQual'].map(bsmtqual_mapping).astype('category')

# map the values after encoding and convert to categorical
X_test['BsmtCond'] = X_test['BsmtCond'].map(bsmtcond_mapping).astype('category')

# Replace missing values with 'NA' (No Basement)
X_test['BsmtExposure'] = X_test['BsmtExposure'].fillna('NA')

# map the values after encoding and convert to categorical
X_test['BsmtExposure'] = X_test['BsmtExposure'].map(bsmtexposure_mapping).astype('category')

# Replace missing values with 'NA'
X_test['BsmtFinType1'] = X_test['BsmtFinType1'].fillna('NA')

# map the values after encoding and convert to categorical
X_test['BsmtFinType1'] = X_test['BsmtFinType1'].map(bsmtfintype1_mapping).astype('category')

# map the values after encoding and convert to categorical
X_test['BsmtFinType2'] = X_test['BsmtFinType2'].map(bsmtfintype2_mapping).astype('category')


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

# maping the values
X_test['OverallCond'] = X_test['OverallCond'].map(overallcond_mapping)

# One-Hot Encode
heating_ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

heating_test = heating_ohe.fit_transform(
    X_test[['Heating']]
)

# Convert the encoded array to a DataFrame:
heating_test_df = pd.DataFrame(
    heating_test,
    columns=heating_ohe.get_feature_names_out(['Heating']),
    index=X_test.index
)

# Drop the original column:
X_test = X_test.drop(columns=['Heating'])

# Add the encoded columns to X_train data
X_test = pd.concat([X_test, heating_test_df], axis=1)

# map the values after encoding and convert to categorical
X_test['HeatingQC'] = X_test['HeatingQC'].map(heatingqc_mapping).astype('category')

# map the values after encoding and convert to categorical
X_test['CentralAir'] = X_test['CentralAir'].map(centralair_mapping).astype('category')

# One-Hot Encoding
electrical_ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

electrical_test = electrical_ohe.fit_transform(
    X_test[['Electrical']]
)

# Convert to categorical variable
X_test['Electrical'] = X_test['Electrical'].astype('category')

# Convert to a DataFrame:
electrical_test_df = pd.DataFrame(
    electrical_test,
    columns=electrical_ohe.get_feature_names_out(['Electrical']),
    index=X_test.index
)

# Drop the original column:
X_test = X_test.drop(columns=['Electrical'])

# Add the encoded columns to X_train data
X_test = pd.concat([X_test, electrical_test_df], axis=1)

# Handling missing values

X_test["GarageYrBlt"] = X_test["GarageYrBlt"].fillna(0)

# compute median value for LotFrontage
lotfrontage_median = X_test["LotFrontage"].median()

# replace all na values with median score
X_test["LotFrontage"] = X_test["LotFrontage"].fillna(lotfrontage_median)

X_test["MasVnrArea"] = X_test["MasVnrArea"].fillna(0)










print(X_test.info())
print(X_test.columns)
print(X_train.columns)
print("The shape of training data is :", X_train.shape)
print("The shape of testing data is :", X_test.shape)


###############################################
######## Regression Random Forest Model########
###############################################

# keep a copy of the test IDs
test_ids = X_test['Id'].copy()

# Drop Id column from X_tain and X_test
X_train = X_train.drop(columns=['Id'])
X_test = X_test.drop(columns=['Id'])


# print the shape of the data
print("The shape of training data is :", X_train.shape)
print("The shape of testing data is :", X_test.shape)

# tain_test split
X = X_train.drop(columns=["SalePrice"])
y = X_train["SalePrice"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

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

# Evaluate the model
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = root_mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")













