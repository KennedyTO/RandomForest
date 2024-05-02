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

### Grand-mean centering Level 1 predictors ###
# Check variable types and only center numeric variables
# THIS IS STEP 0 in the Sommet & Morselli (2021) Instruction

level_1_predictors <- c("ENVAWARE", "EPIST", "ESCS", "PARED", "IMMIG", "REPEAT", "SCIEEFF", 
                        "ST022Q01TA", "WEALTH", "HEDRES", "HOMEPOS", "ICTRES")

# Applying grand-mean centering to each numeric predictor and skip non-numeric
for(predictor in level_1_predictors) {
  if (is.numeric(df[[predictor]])) {
    df[[paste0(predictor, "_centered")]] <- df[[predictor]] - mean(df[[predictor]], na.rm = TRUE)
    print(paste(predictor, "has been centered."))
  } else {
    print(paste(predictor, "is not numeric and has not been centered."))
  }
}

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

######## MLM with df_train dataset ######## 

### Hypothesis ###
# Student-Level Influences: Students' science performance (PV1SCIE) is 
# positively influenced by their environmental awareness (ENVAWARE).
# School-Level Effects on Science Performance:
# Schools with better leadership (LEAD) have students with higher average science performance, 
# indicating the impact of school governance on educational outcomes.
# Combined Influences
# The science performance of students (PV1SCIE) is positively associated with their 
# environmental awareness (ENVAWARE), and this relationship is further influenced by 
# the quality of school leadership (LEAD).


# Start capturing all output to a file
# sink("model_output.txt")

### STEP 1: BUILDING AN EMPTY MODEL TO CALCULATE THE ICC/DEFF ###
library(lme4)

# Empty model with random intercepts for schools
empty_model <- lmer(PV1SCIE ~ 1 + (1 | CNTSCHID), data = df_train)

# Calculate ICC
var_random <- as.numeric(VarCorr(empty_model)$CNTSCHID[1])  # Variance of random effects (between schools)
var_resid <- attr(VarCorr(empty_model), "sc")^2             # Residual variance (within schools)
icc <- var_random / (var_random + var_resid)

# Print ICC
print(paste("ICC:", icc))

# Calculate DEFF
n <- mean(table(df$CNTSCHID))  # Average number of students per school
deff <- 1 + (n - 1) * icc

# Print DEFF
print(paste("DEFF:", deff))

# Fit the model with no randome slopes
library(lme4)

# Constrained Intermediate Model: Fixed effects for predictors
constrained_model <- lmer(PV1SCIE ~ ENVAWARE + LEAD + (1 | CNTSCHID), data = df_train)
summary(constrained_model)

# Fit the model with random slopes
augmented_model <- lmer(PV1SCIE ~ ENVAWARE + LEAD + (1 + ENVAWARE + LEAD | CNTSCHID), data = df_train)
summary(augmented_model)

# Calculate Deviance constrained - Deviance augmented
# Likelihood Ratio Test
anova(constrained_model, augmented_model)

# Calculate R-squared values
library(performance)
r_squared <- r2(augmented_model)
print(r_squared)

# Predictions from the model
predictions <- predict(augmented_model, re.form = NULL)  # re.form = NULL to include random effects

# Observed values
observed <- df_train$PV1SCIE

# Calculate RMSE
rmse_value <- sqrt(mean((predictions - observed)^2))
print(paste("RMSE:", rmse_value))

# Calculate MAE
mae_value <- mean(abs(predictions - observed))
print(paste("MAE:", mae_value))

### STEP 3.1 - BUILDING THE FINAL MODEL ###
library(lme4)
library(performance)

# Final model with random slopes for ENVAWARE and LEAD
final_model <- lmer(PV1SCIE ~ ENVAWARE + LEAD + (1 + ENVAWARE + LEAD | CNTSCHID), data = df_train)

# Summary of the final model to look at coefficients and confidence intervals
summary(final_model)

# Obtain AIC of the final_model
AIC(final_model)

# Obtain BIC of the final_model
BIC(final_model)

# Confidence intervals for the fixed effects
conf_int <- confint(final_model, level = 0.95)
print(conf_int)

### CIs indicated that the Final Model needs to be simplified. ###

### STEP 3.2 Simplified model ###
# Define the formula for the simplified model
formula_final <- PV1SCIE ~ ENVAWARE + LEAD + (1 | CNTSCHID) + 
  (0 + ENVAWARE | CNTSCHID) + (0 + LEAD | CNTSCHID)

# Build the simplified model using lmer
simplified_final_model <- lmer(formula_final, data = df_train)

# Summary of the simplified final model
summary(simplified_final_model)

# Obtain AIC
AIC(simplified_final_model)

# Obtain BIC
BIC(simplified_final_model)

# Confidence intervals for the fixed effects
conf_int <- confint(simplified_final_model)
print(conf_int)

# Calculate R^2, MAE, RMSE values for the simplified final model
r_squared_simplified <- performance::r2(simplified_final_model)
print(paste("R-squared for the simplified final model:", r_squared_simplified))

# Predictions from the simplified final model
predictions_simplified <- predict(simplified_final_model)  # Default includes all effects

# Observed values
observed_simplified <- na.omit(df_train$PV1SCIE)  

# Ensure matched lengths after NA omission
if(length(observed_simplified) != length(predictions_simplified)){
  predictions_simplified <- predictions_simplified[!is.na(df_train$PV1SCIE)]
}

# Calculate RMSE for the simplified final model
rmse_simplified <- sqrt(mean((predictions_simplified - observed_simplified)^2))
print(paste("RMSE for the simplified final model:", rmse_simplified))

# Calculate MAE for the simplified final model
mae_simplified <- mean(abs(predictions_simplified - observed_simplified))
print(paste("MAE for the simplified final model:", mae_simplified))

# Plotting diagnostics
par(mfrow = c(2, 2))
plot(simplified_final_model)

### Cross Validate the Model ###
# Setup cross-validation (10-fold) within the training dataset
set.seed(1)  # For reproducibility
folds <- createFolds(df_train$PV1SCIE, k = 5, list = TRUE, returnTrain = TRUE)

# Initialize a list to store results for each fold
results <- list()

# Loop through each fold
for(i in seq_along(folds)) {
  fold_train <- df_train[folds[[i]], ]
  fold_test <- df_train[-folds[[i]], ]

  # Get the levels of CNTSCHID present in the training subset
  train_levels <- unique(fold_train$CNTSCHID)

  # Filter out rows in the test set with new levels of CNTSCHID
  fold_test <- fold_test[fold_test$CNTSCHID %in% train_levels, ]

  # Proceed if there are any data left in the test set
  if(nrow(fold_test) > 0) {
    model <- lmer(formula_final, data = fold_train, 
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2000)))
  
    predictions <- predict(model, newdata = fold_test, re.form = NULL)  # Include random effects
  
    rmse <- sqrt(mean((predictions - fold_test$PV1SCIE)^2))
    mae <- mean(abs(predictions - fold_test$PV1SCIE))
  
    ss_total <- sum((fold_test$PV1SCIE - mean(fold_test$PV1SCIE))^2)
    ss_res <- sum((fold_test$PV1SCIE - predictions)^2)
    r_squared <- 1 - (ss_res / ss_total)
  
    results[[i]] <- list(RMSE = rmse, MAE = mae, R2 = r_squared)
  } else {
    results[[i]] := list(RMSE = NA, MAE = NA, R2 = NA)
  }
}

# Calculate average RMSE, MAE, and R^2 across all folds, properly handling NA values
avg_rmse <- mean(sapply(results, function(x) x$RMSE), na.rm = TRUE)
avg_mae <- mean(sapply(results, function(x) x$MAE), na.rm = TRUE)
avg_r_squared <- mean(sapply(results, function(x) x$R2), na.rm = TRUE)

# Print average results
print(paste("Average RMSE:", avg_rmse))
print(paste("Average MAE:", avg_mae))
print(paste("Average R-squared:", avg_r_squared))
