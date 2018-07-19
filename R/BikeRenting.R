rm(list=ls())
library(MLmetrics)
library(Metrics)
library(rsq)
library(caret)

# ================
# LOADING DATASET
# ================
train_test = read.csv("./Dataset/day.csv")


# ===============
# EXPLORING DATA
# ===============
summary(train_test)
str(train_test)
lapply(train_test, FUN = function(x) length(unique(x)))
lapply(train_test, FUN = function(x) sum(is.na(x)))
lapply(train_test, FUN = function(x) sum(is.null(x)))


# =================================
# PRE-PROCESSING AND DATA CLEANING
# =================================
train_test$dteday = as.character(train_test$dteday)
train_test$day = lapply(X = train_test$dteday, FUN = function(x) strsplit(x = x, split = "-")[[1]][3])
train_test$day = as.integer(train_test$day)
train_test$dteday = NULL
train_test$instant = NULL
train_test = train_test[, c("season", "yr", "mnth", "day", "weekday", "workingday", "weathersit", "temp", 
                   "atemp", "hum", "windspeed", "casual", "registered", "cnt")]


# =================
# OUTLIER ANALYSIS
# =================
boxplot(train_test$windspeed, main="windspeed")
outlier_values = boxplot.stats(train_test$windspeed)$out
train_test[train_test$windspeed %in% outlier_values, ][, 'windspeed'] = NA
library(DMwR)
train_test = knnImputation(data = train_test, k = 3)
cat(c("Total Missing Values = ", sum(is.na(train_test$windspeed))))


# ==================
# FEATURE SELECTION
# ==================

# Corrplot
library(corrplot)
c = cor(train_test)
corrplot(c, method = "number")

# lm
a = lm(cnt~., data = train_test)
summary(a)

# Dropping unimportant columns
train_test$mnth = NULL
train_test$atemp = NULL
train_test$casual = NULL
train_test$day = NULL
train_test$hum = NULL


lapply(X = train_test, FUN = function(x) length(unique(x)))
str(train_test)
for(i in seq(1:5)){
  train_test[, i] = factor(train_test[, i])
}
train_test$registered = as.numeric(train_test$registered)
train_test$cnt = as.numeric(train_test$cnt)


# ================
# FEATURE SCALING
# ================

# Normalization
# --------------
par(mfrow=c(2,4))
par(col.lab="red")
# par(mar=c(5,2,8,1))
# before normalization
for(x in colnames(train_test[,6:9])){
  hist(train_test[, x], main = "Before Normalization", xlab = x, ylim = c(0,300))
  abline(h = 1000, lty=3)
}
# after normalization
for(x in colnames(train_test[,6:9])){
  train_test[, x] = (train_test[, x] - min(train_test[, x]))/(max(train_test[, x])-min(train_test[, x]))
  hist(train_test[, x], main = "After Normalization", xlab = x, col = "green", ylim = c(0,300))
  abline(h = 1000, lty=3)
}


# Standardization
# ----------------
par(mfrow=c(2,4))
# par(mar=c(5,3,2,1))
for(x in colnames(train_test[,6:9])){
  if(is.numeric(train_test[, x])){
    qqnorm(train_test[, x], xlab = x, main = x)
    qqline(train_test[, x])
    hist(train_test[, x], xlab = x, main = x)
  }
}
train_test_for_standardization_visualization = train_test
for(x in colnames(train_test[,6:9])){
  train_test[, x] = (train_test[, x] - mean(train_test[, x]))/(sd(train_test[, x]))
}
# Visualization of graph after standardization
par(mfrow=c(4,2))
par(mar=c(5,5,1,1)+0.1)
for(x in colnames(train_test[,6:9])){
  hist(train_test_for_standardization_visualization[, x], xlab = x, main = "Before Standardization", ylim=c(0,300), xlim = c(-4,4))
  hist(train_test[, x], xlab = x, main = "After Standardization", col = "green", ylim=c(0,300), xlim = c(-4,4))
}


# ==============================
# SPLITTING INTO TRAIN AND TEST
# ==============================
index = sample(x = 1:nrow(train_test), size = 0.7*nrow(train_test))
train = train_test[index, ]
test = train_test[-index, ]


# ===============
# MODEL BUILDING
# ===============
  
model_name = vector()  
rsq_value = vector()
rmse_value = vector()
mae_value = vector()

register_model_with_scores = function(name, rsq, rmse, mae){
  model_name <<- append(model_name, name)
  rsq_value <<- append(rsq_value, rsq)
  rmse_value <<- append(rmse_value, rmse)
  mae_value <<- append(mae_value, mae)
} 

print_score <- function(postResample_score, model_name, register_model=1){
  rsq = postResample_score[[2]]
  mae = postResample_score[[3]]
  rmse_val = postResample_score[[1]]
  cat(c("R-sq = ", rsq, "\n"))
  cat(c("RMSE = ", rmse_val, "\n"))
  cat(c("MAE = ", mae, "\n"))
  if(register_model==1){
    register_model_with_scores(model_name, rsq, rmse_val, mae)
  }
}
  
  
# Linear Regression
# ==================
name1 = "Linear Regression"
linreg = lm(formula = cnt~., data = train)
summary(linreg)
# plotting graph
par(mfrow=c(1,1))
scatter.smooth(train$registered, train$cnt, col='chartreuse3')
lines(loess.smooth(train$registered, train$cnt), col="red", lwd=2)
# predicting values
pred1 = predict(object = linreg, newdata = test[, -11])
score1 = postResample(pred1, test$cnt)
print_score(score1, name1)




# GBM
# ====
name2 = "GBM"
library(iterators)
library(parallel)
library(doMC)
library(caret)
set.seed(222)
## detectCores() returns 16 cpus
registerDoMC(16)
## Set up caret model training parameters
CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv", number = 5, repeats = 5, 
                                 verboseIter = FALSE, allowParallel = TRUE)
gbmFit <- train(cnt ~ ., method = "gbm", metric = "RMSE", maximize = FALSE, 
                trControl = CARET.TRAIN.CTRL, 
                tuneGrid = expand.grid(n.trees = (4:10) * 50, interaction.depth = c(5), 
                                       shrinkage = c(0.05), n.minobsinnode = c(10)), 
                data = train, verbose = FALSE)
print(gbmFit)

## Predictions
pred2 <- predict(gbmFit, newdata = test[, -9])
score2 = postResample(pred2, test$cnt)
print_score(score2, name2, register_model = 1)


# XGBOOST
# ========
name3 = "XGBoost"
library(xgboost)

xgbFit = xgboost(data = data.matrix(train[, -9]), nfold = 5, label = data.matrix(train$cnt), print_every_n = 100,
                 nrounds = 1000, verbose = T, objective = "reg:linear", eval_metric = c("rmse"))
# Predictions
pred3 <- predict(xgbFit, newdata = data.matrix(test[, -9]))
score3 = postResample(pred3, test$cnt)
print_score(score3, name3, register_model = 0)


# XGBoost Parameter Tuning
# ========================

# tuning eta value
# ----------------
para_name = "eta"
secq = seq(0.01, 0.3, 0.01)
score_list = vector()
for (eta_val in secq){
  xgbFit = xgboost(data = data.matrix(train[, -9]), nfold = 5, label = data.matrix(train$cnt), print_every_n = 100,
                   nrounds = 1000, verbose = T, objective = "reg:linear", eval_metric = c("rmse"), 
                   early_stopping_rounds = 20, eta = eta_val)
  # Predictions
  pred <- predict(xgbFit, newdata = data.matrix(test[, -9]))
  score = postResample(pred, test$cnt)
  score_list  = append(score_list, score[[2]])
}
j=1
best_score_index = NULL
for (item in score_list){
  print(item)
  if(item == max(score_list)){
    best_score_index = j
  }
  j = j + 1
}
k=1
for (eta_val in secq){
  if (k==best_score_index){
    best_eta = secq[k]
  }
  k = k + 1
}
cat(c("Best ", para_name, " value : ", best_eta))


# tuning max_depth value
# -----------------
para_name = "max_depth"
secq = seq(1, 10, 1)
score_list = vector()
for (val in secq){
  xgbFit = xgboost(data = data.matrix(train[, -9]), nfold = 5, label = data.matrix(train$cnt), print_every_n = 100,
                   nrounds = 1000, verbose = T, objective = "reg:linear", eval_metric = c("rmse"), 
                   early_stopping_rounds = 20, eta = best_eta, max_depth=val)
  # Predictions
  pred <- predict(xgbFit, newdata = data.matrix(test[, -9]))
  score = postResample(pred, test$cnt)
  score_list  = append(score_list, score[[2]])
}
j=1
best_score_index = NULL
for (item in score_list){
  print(item)
  if(item == max(score_list)){
    best_score_index = j
  }
  j = j + 1
}
k=1
best_max_depth = NULL
for (val in secq){
  if (k==best_score_index){
    best_max_depth = secq[k]
  }
  k = k + 1
}
cat(c("Best ", para_name, " value : ", best_max_depth))
cat(c("Achieved R-sq : ", score_list[best_score_index]))


# tuning gamma value
# -----------------
para_name = "gamma"
secq = seq(0.001, 0.02, 0.01)
score_list = vector()
for (val in secq){
  xgbFit = xgboost(data = data.matrix(train[, -9]), nfold = 5, label = data.matrix(train$cnt), print_every_n = 100,
                   nrounds = 1000, verbose = T, objective = "reg:linear", eval_metric = c("rmse"), 
                   early_stopping_rounds = 20, eta = best_eta, max_depth=best_max_depth, gamma= val)
  # Predictions
  pred <- predict(xgbFit, newdata = data.matrix(test[, -9]))
  score = postResample(pred, test$cnt)
  score_list  = append(score_list, score[[2]])
}
j=1
best_score_index = NULL
for (item in score_list){
  print(item)
  if(item == max(score_list)){
    best_score_index = j
  }
  j = j + 1
}
k=1
best_gamma =NULL
for (val in secq){
  if (k==best_score_index){
    best_gamma = secq[k]
  }
  k = k + 1
}
cat(c("Best ", para_name, " value : ", best_gamma))
cat(c("Achieved R-sq : ", score_list[best_score_index]))

# Final run with all the best parameter values
set.seed(123)
xgbFit = xgboost(data = data.matrix(train[, -9]), nfold = 5, label = data.matrix(train$cnt), print_every_n = 100,
                 nrounds = 1000, verbose = T, objective = "reg:linear", eval_metric = c("rmse"), 
                 eta = best_eta, gamma = best_gamma, max_depth = best_max_depth)
# Predictions
pred3 <- predict(xgbFit, newdata = data.matrix(test[, -9]))
score3 = postResample(pred3, test$cnt)
print_score(score3, name3, register_model = 1)



# Support Vector Machine
# =======================
name4 = "SVM"
library(e1071)
svm_mod <- svm(cnt ~ ., data=train, type="nu-regression", kernel="radial")
pred4 <- predict(svm_mod, newdata = test[, -9])
score4 = postResample(pred4, test$cnt)
print_score(score4, name4, register_model = 0)
#Plot
plot(test$cnt,col = "blue", pch=17)
points(pred4, col = "green", pch=17)
legend('topleft',legend=c("Actual Values", "Predicted Values"), col=c("blue", "green"), lty=1:2, cex=0.8)

# SVM Parameter Tuning
# =====================

# tuning gamma value
# -----------------
para_name = "gamma"
secq = seq(0.001, 0.01, 0.001)
score_list = vector()
for (val in secq){
  svm_mod <- svm(cnt ~ ., data=train, type="nu-regression", kernel="radial", gamma=val)
  # Predictions
  pred <- predict(svm_mod, newdata = test[, -9])
  score = postResample(pred, test$cnt)
  score_list  = append(score_list, score[[2]])
}
j=1
best_score_index = NULL
for (item in score_list){
  print(item)
  if(item == max(score_list)){
    best_score_index = j
  }
  j = j + 1
}
k=1
best_svm_gamma = NULL
for (val in secq){
  if (k==best_score_index){
    best_svm_gamma = secq[k]
  }
  k = k + 1
}
cat(c("Best ", para_name, " value : ", best_svm_gamma))
cat(c("Achieved R-sq : ", score_list[best_score_index]))


# tuning cost value
# -----------------
para_name = "cost"
secq = seq(25, 50, 1)
score_list = vector()
for (val in secq){
  svm_mod <- svm(cnt ~ ., data=train, type="nu-regression", kernel="radial", gamma=best_svm_gamma, cost=val)
  # Predictions
  pred <- predict(svm_mod, newdata = test[, -9])
  score = postResample(pred, test$cnt)
  score_list  = append(score_list, score[[2]])
}
j=1
best_score_index = NULL
for (item in score_list){
  print(item)
  if(item == max(score_list)){
    best_score_index = j
  }
  j = j + 1
}
k=1
best_svm_cost = NULL
for (val in secq){
  if (k==best_score_index){
    best_svm_cost = secq[k]
  }
  k = k + 1
}
cat(c("Best ", para_name, " value : ", best_svm_cost))
cat(c("Achieved R-sq : ", score_list[best_score_index]))


# Final run with all the best parameter values
set.seed(123)
svm_mod <- svm(cnt ~ ., data=train, type="nu-regression", kernel="radial", gamma=best_svm_gamma, cost=best_svm_cost)
# Predictions
pred4 <- predict(svm_mod, newdata = test[, -9])
score4 = postResample(pred4, test$cnt)
print_score(score4, name4, register_model = 1)


# ======================
# MODEL'S SCORE SUMMARY
# ======================
models_df = data.frame(model_name, rsq_value, rmse_value, mae_value)
models_df


# ===========
# CONCLUSION
# ===========

# From the above summary of the models we can conclude that XGBoost regressor
# is performing better as compared to other models.This conclusion is made on the basis 
# of R-square, mean absolute error(MAE) and root mean square error(RMSE) values.

