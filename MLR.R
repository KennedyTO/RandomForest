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

# Get the number of levels in the SCHLTYPE variable
nlevels(df$SCHLTYPE)
nlevels(df$IMMIG)
nlevels(df$CNTSCHID)

# Identify number of rows and columns in the dataset
dim(df)

# Find the missing values in the dataset
missing_values <- colSums(is.na(df))

# Output the missing values
print(missing_values)

### List of Variables for the MLM Model ###
#### Level 1: Student-Level Variables ####
##These are the attributes specific to individual students:
#PV1SCIE: Science performance (dependent variable).
#ENVAWARE: Awareness of environmental matters.
#EPIST: Epistemic beliefs about science.
#ESCS: Economic, social, and cultural status.
#PARED: Parental education level.
#IMMIG: Immigration status.
#REPEAT: Whether the student has repeated a grade.
#SCIEEFF: Science self-efficacy.
#ST022Q01TA: Language of the test taken.
#WEALTH: Family wealth possessions.
#HEDRES: Educational resources at home.
#HOMEPOS: Home possessions.
#ICTRES: Information and communication technology resources at home.

#### Level 2: School-Level Variables ####
#These describe characteristics of the schools that students attend:
#LEAD: School leadership quality.
#SCHSIZE: School size.
#TOTAT: Total number of teachers at the school.
#GRADE: Grade compared to modal grade in country.
#PROSTAT: Professional status (could refer to the status of teaching staff if related to school context).
#TOTST: Total number of students at the school.
#SCHLTYPE: Type of school (public, private, etc.).
#CLSIZE: Class size.
#CNTSCHID: Identifier for the school, which is key for random effects modeling to account for school-level variation.

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
                    "PROSTAT", "TOTST", "WEALTH", "SCHLTYPE", "CLSIZE", "CNTSCHID")

df22 <- select(df, all_of(important_vars))
dim(df22)

# Split data into training (80%) and testing (20%) sets
set.seed(1)
trainIndex <- createDataPartition(df22$PV1SCIE, p = .8, list = FALSE, times = 1)
df_train <- df22[trainIndex, ]
df_test  <- df22[-trainIndex, ]

# Check the dimensions of your training and testing datasets to ensure they're not empty
dim(df_train)
dim(df_test)

# Identify numeric variables from the important_vars list, excluding the outcome variable PV1SCIE
numeric_vars <- important_vars[-1][sapply(df_train[, ..important_vars[-1]], is.numeric)]
print(numeric_vars)


#### Building the MRL model ####
### This method ended up illegible. Change the course of action to heatmap ###
# Load the necessary libraries
library(ggplot2)
library(reshape2)

# Ensure that you have the correct numeric columns from your dataframe
# First, let's clearly identify the numeric columns
numeric_vars <- names(df_train)[sapply(df_train, is.numeric)]

# Subset the dataframe to only include these numeric columns using the .. to refer to external variable
numeric_data <- df_train[, ..numeric_vars]

# Now calculate the correlation matrix
correlation_matrix <- cor(numeric_data)

# Print the correlation matrix to view it
print(correlation_matrix)

# Melt the correlation matrix for visualization
melted_correlation <- melt(correlation_matrix)

# Create a heatmap of the correlation matrix
ggplot(melted_correlation, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(midpoint = 0, low = "deepskyblue4", high = "deepskyblue4", mid = "white") +
  theme_minimal() +
  labs(fill = "Correlation") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# For reference, obtain the VIF values for the model
# Load the necessary libraries
library(car) 

# Ensure that you have the correct numeric columns from your dataframe
numeric_vars <- names(df_train)[sapply(df_train, is.numeric)]

# Convert the subset of the data.table to a data frame
numeric_data_df <- as.data.frame(df_train[, ..numeric_vars])

# Fit the model on the numeric data
Model_VIF <- lm(PV1SCIE ~ ., data = numeric_data_df)

# Calculate VIF
vif_values <- vif(Model_VIF)

# Print the VIF values
print(vif_values)

### MLR without potentially problematic multicollinear variables ###
### Variables with ONLY VIF < 5 were considered in the model) ###

# Identify variables with VIF < 5 to keep in the model
keep_vars <- names(vif_values)[vif_values < 5]

#print variable names above < 5
print(keep_vars)

# Filter the data to only include the variables with VIF < 5
filtered_data <- as.data.frame(df_train[, c("PV1SCIE", keep_vars), with = FALSE])

# Build the Multiple Linear Regression (MLR) model using the filtered data
final_model <- lm(PV1SCIE ~ ., data = filtered_data)

# Print the summary of the model to see the results
summary(final_model)

# Print the confidence intervals for the model
confint(final_model)

# Check the assumptions
# Check for normality of residuals

par(mfrow = c(2, 2))
plot(final_model)

# Calculate R2, RMSE, and MAE before the cross-validation
predictions <- predict(final_model, newdata = df_test)
results <- data.frame(Predicted = predictions, Actual = df_test$PV1SCIE)

# Calculate the R-squared value for the model
r_squared_reduced <- cor(results$Predicted, results$Actual)^2


# Calculate the MAE for the model
mae_reduced <- mean(abs(results$Predicted - results$Actual))


# Calculate the RMSE for the model
rmse_reduced <- sqrt(mean((results$Predicted - results$Actual)^2))
print(c(R_squared = r_squared_reduced, MAE = mae_reduced, RMSE = rmse_reduced))

#### Cross-validate the VIF<5 model with 5-fold ####
library(caret)
set.seed(1)
cv_results <- train(PV1SCIE ~ ., data = filtered_data, method = "lm", 
                    trControl = trainControl(method = "cv", number = 5))

# Output the cross-validation results
print(cv_results)


### Test the Model ###
# Test the model on the testing dataset
predictions <- predict(final_model, newdata = df_test)
results <- data.frame(Predicted = predictions, Actual = df_test$PV1SCIE)

# Calculate the R-squared value for the model
r_squared_test <- cor(results$Predicted, results$Actual)^2

# Calculate the MAE for the model
mae_test <- mean(abs(results$Predicted - results$Actual))

# Calculate the RMSE for the model
rmse_test <- sqrt(mean((results$Predicted - results$Actual)^2))

print(c(RMSE_test = rmse_test, R2_test = r_squared_test, MAE_test = mae_cv))

#### Build a MLR without considering VIF (Ignoring the Multicolliearity) ####
# Build a full MLR model without considering VIF
set.seed(1)
Full_MLR <- lm(PV1SCIE ~ ., data = df_train)
summary(Full_MLR)

# Calculate R2, RMSE, and MAE before the cross-validation
full_predictions <- predict(Full_MLR, newdata = df_test)
full_results <- data.frame(Predicted = full_predictions, Actual = df_test$PV1SCIE)

# Calculate the R-squared value for the model
full_r_squared <- cor(full_results$Predicted, full_results$Actual)^2
print(full_r_squared)

# Calculate the MAE for the model
full_mae <- mean(abs(full_results$Predicted - full_results$Actual))
print(full_mae)

# Calculate the RMSE for the model
full_rmse <- sqrt(mean((full_results$Predicted - full_results$Actual)^2))
print(full_rmse)

### Cross-validate the full MLR model with 5-folds ###
# Cross-validate the full MLR model with 5-folds
set.seed(1)
cv_results <- train(PV1SCIE ~ ., data = df_train, method = "lm", 
                    trControl = trainControl(method = "cv", number = 5))

# Output the cross-validation results
print(cv_results)

# Test the model on the testing dataset
predictions <- predict(Full_MLR, newdata = df_test)
results <- data.frame(Predicted = predictions, Actual = df_test$PV1SCIE)


# Calculate the RMSE for the model
rmse <- sqrt(mean((results$Predicted - results$Actual)^2))
print(rmse)

# Calculate the R-squared value for the model
r_squared <- cor(results$Predicted, results$Actual)^2

# Calculate the MAE for the model
mae <- mean(abs(results$Predicted - results$Actual))
print(mae)

# Calculate the R-squared value for the model
r_squared <- cor(results$Predicted, results$Actual)^2
print(r_squared)

