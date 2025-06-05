# Libraries ----
library(Boruta)
library(xgboost)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(caret)

#------------------------------------------------------------------#
# XGB-Boruta feature selection technique ------
#------------------------------------------------------------------#

df.extraction <- read.csv("dataAnalysis/data.csv")
df.extraction <- df.extraction %>% rename(TPC = Total.Polyphenol.Content)
df.extraction$pH <- as.numeric(df.extraction$pH)
drop_na(df.extraction)
df.extraction <- drop_na(df.extraction)

X <- as.matrix(df.extraction[, -8])
y <- as.numeric(df.extraction$TPC) - 1

dtrain <- xgb.DMatrix(data = X, label = y)
params <- list(objective = "reg:squarederror")
xgb_model <- xgb.train(params, dtrain, nrounds = 500)

importance_matrix <- xgb.importance(model = xgb_model)

boruta_result <- Boruta(x = X, y = y, doTrace = 2)

final_decision <- TentativeRoughFix(boruta_result)
print(final_decision)

#------------------------------------------------------------------#
# Checking Pearson Correlation between variables ------
#------------------------------------------------------------------#

M <- cor(df.extraction)
corrplot(M, method="number", type="upper")

#------------------------------------------------------------------#
# Machine Learning models to correlate the RGB, HSV, pH and TPC ------
#------------------------------------------------------------------#

# GradientBoostingRegressor, Polynomial, KernelRidge, Elastic Net, XGBoost and Neural Network

set.seed(123)
index <- createDataPartition(df.extraction$TPC, p = 0.8, list = FALSE)
train_data <- df.extraction[index, ]
test_data <- df.extraction[-index, ]

y_test <- test_data$TPC

# Control parameters for cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

# Initialize metrics data frame
metrics_df <- data.frame(Model = character(),
                         RMSE = numeric(),
                         RRMSE = numeric(),
                         KGE = numeric(),
                         stringsAsFactors = FALSE)

## 1. Gradient Boosting Regressor with Grid Search -----
gb_grid <- expand.grid(
  n.trees = c(100, 200, 300),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(5, 10)
)

gb_model <- train(
  TPC ~ R + H + V + S + G + B,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gb_grid,
  verbose = FALSE
)

gb_pred <- predict(gb_model, test_data)

# Removing outliers
data <- data.frame(Observed = y_test,
                   Predicted = gb_pred)

data$Difference <- abs(data$Observed - data$Predicted)
threshold <- mean(data$Difference) + 2 * sd(data$Difference)
data_clean <- data[data$Difference <= threshold, ]

# Calculate metrics
rmse_gb <- RMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
rrmse_gb <- RRMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
kge_gb <- KGE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "Gradient Boosting", RMSE = rmse_gb, RRMSE = rrmse_gb, KGE = kge_gb))

# Graph
graph_gb <- ggplot(mapping = aes(x = y_test, y = gb_pred))+
  geom_point(color = "blue", size = 3)+
  geom_abline(slope=1, color="Red", linewidth=2)+
  theme_minimal()+
  scale_x_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  scale_y_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  xlab("Observed TPC (mg GAE/L)")+
  ylab("Predicted TPC (mg GAE/L)")

## 2. Polynomial Regression with degree optimization ----
poly_model <- lm(TPC ~ poly(R,2) + poly(H,2) + poly(V,2) + poly(S,2) + poly(G,2) + poly(B,2), data = train_data)

poly_pred <- predict(poly_model, test_data)

# Removing outliers
data <- data.frame(Observed = y_test,
                   Predicted = poly_pred)

data$Difference <- abs(data$Observed - data$Predicted)
threshold <- mean(data$Difference) + 2 * sd(data$Difference)
data_clean <- data[data$Difference <= threshold, ]

# Calculate metrics
rmse_poly <- RMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
rrmse_poly <- RRMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
kge_poly <- KGE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "Polynomial Regression", RMSE = rmse_poly, RRMSE = rrmse_poly, KGE = kge_poly))

graph_poly <- ggplot(mapping = aes(x = y_test, y = poly_pred))+
  geom_point(color = "blue", size = 3)+
  geom_abline(slope=1, color="Red", linewidth=2)+
  theme_minimal()+
  scale_x_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  scale_y_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  xlab("Observed TPC (mg GAE/L)")+
  ylab("Predicted TPC (mg GAE/L)")

## 3. Kernel Ridge with Grid Search -----
krr_grid <- expand.grid(
  C = c(0.1, 1, 10),
  sigma = c(0.1, 0.5, 1)
)

krr_model <- train(
  x = as.matrix(train_data[, c("R", "H", "V", "S", "G", "B")]),
  y = train_data$TPC,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = krr_grid
)

krr_pred <- predict(krr_model, as.matrix(test_data[, c("R", "H", "V", "S", "G", "B")]))

# Removing outliers
data <- data.frame(Observed = y_test,
                   Predicted = krr_pred)

data$Difference <- abs(data$Observed - data$Predicted)
threshold <- mean(data$Difference) + 2 * sd(data$Difference)
data_clean <- data[data$Difference <= threshold, ]

# Calculate metrics
rmse_krr <- RMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
rrmse_krr <- RRMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
kge_krr <- KGE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "Kernel Ridge", RMSE = rmse_krr, RRMSE = rrmse_krr, KGE = kge_krr))

# Graph
graph_krr <- ggplot(mapping = aes(x = y_test, y = krr_pred))+
  geom_point(color = "blue", size = 3)+
  geom_abline(slope=1, color="Red", linewidth=2)+
  theme_minimal()+
  scale_x_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  scale_y_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  xlab("Observed TPC (mg GAE/L)")+
  ylab("Predicted TPC (mg GAE/L)")

## 4. Elastic Net with Grid Search -----
elastic_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = 10^seq(-3, 3, length = 100)
)

elastic_model <- train(
  x = as.matrix(train_data[, c("R", "H", "V", "S", "G", "B")]),
  y = train_data$TPC,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = elastic_grid
)

elastic_pred <- predict(elastic_model, as.matrix(test_data[, c("R", "H", "V", "S", "G", "B")]))

# Removing outliers
data <- data.frame(Observed = y_test,
                   Predicted = elastic_pred)

data$Difference <- abs(data$Observed - data$Predicted)
threshold <- mean(data$Difference) + 2 * sd(data$Difference)
data_clean <- data[data$Difference <= threshold, ]

# Calculate metrics
rmse_elastic <- RMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
rrmse_elastic <- RRMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
kge_elastic <- KGE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "Elastic Net", RMSE = rmse_elastic, RRMSE = rrmse_elastic, KGE = kge_elastic))

# Graph
graph_elastic <- ggplot(mapping = aes(x = y_test, y = elastic_pred))+
  geom_point(color = "blue", size = 3)+
  geom_abline(slope=1, color="Red", linewidth=2)+
  theme_minimal()+
  scale_x_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  scale_y_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  xlab("Observed TPC (mg GAE/L)")+
  ylab("Predicted TPC (mg GAE/L)")

## 5. XGBoost with Grid Search ------
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1)
)

xgb_model <- train(
  x = as.matrix(train_data[, c("R", "H", "V", "S", "G", "B")]),
  y = train_data$TPC,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbose = FALSE
)

xgb_pred <- predict(xgb_model, as.matrix(test_data[, c("R", "H", "V", "S", "G", "B")]))

# Removing outliers
data <- data.frame(Observed = y_test,
                   Predicted = xgb_pred)

data$Difference <- abs(data$Observed - data$Predicted)
threshold <- mean(data$Difference) + 2 * sd(data$Difference)
data_clean <- data[data$Difference <= threshold, ]

# Calculate metrics
rmse_xgb <- RMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
rrmse_xgb <- RRMSE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)
kge_xgb <- KGE(data = data_clean, obs = data_clean$Observed, pred = data_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "XGBoost", RMSE = rmse_xgb, RRMSE = rrmse_xgb, KGE = kge_xgb))

# Graph
graph_xgb <- ggplot(mapping = aes(x = y_test, y = xgb_pred))+
  geom_point(color = "blue", size = 3)+
  geom_abline(slope=1, color="Red", linewidth=2)+
  theme_minimal()+
  scale_x_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  scale_y_continuous(breaks = (seq(0, 900, 100)), limits = c(0,900))+
  xlab("Observed TPC (mg GAE/L)")+
  ylab("Predicted TPC (mg GAE/L)")

## 6. Neural network ----
# Prepare training data
train_x <- as.matrix(train_data[, c("R", "H", "V", "S", "G", "B")])
train_y <- as.matrix(train_data$TPC)

# Prepare testing data
test_x <- as.matrix(test_data[, c("R", "H", "V", "S", "G", "B")])

# Train the ELM model
elm_model <- elm_train(
  x = train_x,
  y = train_y,
  nhid = 100,    # Number of hidden neurons
  actfun = "sig" # Activation function (e.g., sigmoid)
)

# Predict on the test set
elm_pred <- elm_predict(elm_model, test_x)

# Removing outliers
data_elm <- data.frame(Observed = y_test,
                       Predicted = elm_pred)
data_elm$Difference <- abs(data_elm$Observed - data_elm$Predicted)
threshold_elm <- mean(data_elm$Difference) + 2 * sd(data_elm$Difference)
data_elm_clean <- data_elm[data_elm$Difference <= threshold_elm, ]

# Calculate metrics
rmse_elm <- RMSE(data = data_elm_clean, obs = data_elm_clean$Observed, pred = data_elm_clean$Predicted)
rrmse_elm <- RRMSE(data = data_elm_clean, obs = data_elm_clean$Observed, pred = data_elm_clean$Predicted)
kge_elm <- KGE(data = data_elm_clean, obs = data_elm_clean$Observed, pred = data_elm_clean$Predicted)

# Append to metrics dataframe
metrics_df <- rbind(metrics_df, data.frame(Model = "ELM", RMSE = rmse_elm, RRMSE = rrmse_elm, KGE = kge_elm))

# Graph the results
graph_elm <- ggplot(mapping = aes(x = y_test, y = elm_pred)) +
  geom_point(color = "blue", size = 3) +
  geom_abline(slope = 1, color = "red", linewidth = 2) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, 900, 100), limits = c(0, 900)) +
  scale_y_continuous(breaks = seq(0, 900, 100), limits = c(0, 900)) +
  xlab("Observed TPC (mg GAE/L)") +
  ylab("Predicted TPC (mg GAE/L)")

## Export metrics to CSV ----
write.csv(metrics_df, "dataAnalysis/model_metrics.csv", row.names = FALSE)

## Arrange graph -----
ggarrange(graph_gb, graph_poly, graph_krr, graph_elastic, graph_xgb, graph_elm,
          labels = c("A", "B", "C", "D", "E", "F"),
          ncol = 2, nrow = 3)
