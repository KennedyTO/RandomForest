# Load required libraries
library(data.table)
library(randomForest)
library(caret)
library(dplyr)

# Load data from a CSV file using data.table's fread for faster loading.
df <- fread("~/R_Projects/Thesis_08/completedData.merged.csv")

## Data preparation
# Cleaning/managing variables
df$ST004D01T <- ifelse(df$ST004D01T == "2", 1, 0)# Male = 1, Female = 0
df$ST022Q01TA <- ifelse(df$ST022Q01TA == "2", 1, 0) # Others = 1, Language of test = 0
df$IMMIG <- as.factor(df$IMMIG)
df$SCHLTYPE <- as.factor(df$SCHLTYPE)
df$CNTSCHID <- as.factor(df$CNTSCHID)


# Find the missing values in the dataset
missing_values <- colSums(is.na(df))

# Output the missing values
print(missing_values)

# Identify the numeric columns in the subset of your dataframe:
subset_indices <- c(3, 12:15, 20, 27, 28, 36, 37, 40:45, 48, 50:68)
subset_df <- df[, ..subset_indices]
numeric_columns <- sapply(subset_df, is.numeric)

# Apply the scaling only to these numeric columns:
numeric_column_names <- names(subset_df)[numeric_columns]
subset_df[, (numeric_column_names) := lapply(.SD, scale), .SDcols = numeric_column_names]


# Replace the subset of the original dataframe with the scaled values:
df[, subset_indices] <- subset_df


# Scales the numeric columns in the subset to have a mean of 0 and a standard deviation of 1. 
# This is important because it ensures that all variables are on the same scale, 
# which is necessary for many machine learning algorithms to work properly.
predictors <- names(df[,c(2:25, 27:68)])
outcome <- "PV1SCIE" # This variable means the science performance of students.
f <- as.formula(paste(outcome, paste(predictors, collapse = "+"), sep = "~"))
print(f)


# Subset with important variables
important_vars <- c("PV1SCIE", "ENVAWARE", "EPIST", "ESCS", "LEAD", "PARED", 
                    "SCHSIZE", "TOTAT", "GRADE", "HEDRES", "HOMEPOS", "ICTRES", 
                    "IMMIG", "REPEAT", "SCIEEFF", "ST022Q01TA", 
                    "PROSTAT", "TOTST", "WEALTH", "SCHLTYPE", 
                    "SCHSIZE", "CLSIZE")

df23 <- select(df, all_of(important_vars))

# Split data into training (80%) and testing (20%) sets
set.seed(1)
trainIndex <- createDataPartition(df23$PV1SCIE, p = .8, list = FALSE, times = 1)
df_train <- df23[trainIndex, ]
df_test  <- df23[-trainIndex, ]

# Check the dimensions of your training and testing datasets to ensure they're not empty
dim(df_train)
dim(df_test)

# Identify numeric variables from the important_vars list, excluding the outcome variable PV1SCIE
numeric_vars <- important_vars[-1][sapply(df_train[, ..important_vars[-1]], is.numeric)]

# Model fitting
#### Random Forest Model with df20 dataset ####
set.seed(123)
RF_noCV <- randomForest(PV1SCIE ~., data = df_train)

# Prediction on the Test Set
predictions_noCV <- predict(RF_noCV, df_train)

# Calculate R^2
R2_RF_noCV <- R2(predictions_noCV, df_train$PV1SCIE)

# Calculate RMSE
rmse_RF_noCV <- RMSE(predictions_noCV, df_train$PV1SCIE)

# Calculate MAE
mae_RF_noCV <- MAE(predictions_noCV, df_train$PV1SCIE)

#R2_RF_noCV
rmse_RF_noCV
mae_RF_noCV

# Evaluate the training model
predictions <- predict(model, newdata = df_test)
results <- list(
  R2 = cor(df_test$PV1SCIE, predictions)^2,
  MAE = mean(abs(df_test$PV1SCIE - predictions)),
  RMSE = sqrt(mean((df_test$PV1SCIE - predictions)^2))
)
print(results)

##### Random Forest Model with Cross-Validation ####
# Cross-Validation
set.seed(123)
RF_CV <- train(PV1SCIE ~., data = df_train, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Print results of the cross-validation
RF_CV

# Save the model to a file
saveRDS(RF_CV, "RF_CV.rds")

# Load the model from the file
RF_CV <- readRDS("RF_CV.rds")


### Testing using the test set ###
# Prediction on the Test Set
predictions_CV <- predict(RF_CV, df_test)

# Calculate R^2
R2_RF_CV <- R2(predictions_CV, df_test$PV1SCIE)

# Calculate RMSE
rmse_RF_CV <- RMSE(predictions_CV, df_test$PV1SCIE)

# Calculate MAE
mae_RF_CV <- MAE(predictions_CV, df_test$PV1SCIE)

# Print results together
Print(c(R2 = R2_RF_CV, RMSE = rmse_RF_CV, MAE = mae_RF_CV)))