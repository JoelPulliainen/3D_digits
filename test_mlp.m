clc
clear all
close all

%% Data collection and training set new values

[data,class] = data_collect();
[traindata,trainclass, validationdata,validationclass] = data_splitter(data,class);

%% Test with the validation set
[Predicted_digits,Accuracy] = mlp_test(validationdata,validationclass)

%% Test with  all
[Predicted_digits_all,Accuracy_all] = mlp_test(data,class)