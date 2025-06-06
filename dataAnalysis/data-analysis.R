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