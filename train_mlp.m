clc
clear all
close 


%% Data collection and training set new values

[data,class] = data_collect();
[traindata,trainclass, validationdata,validationclass] = data_splitter(data,class);

%% Train
[weightHidden1,weightHidden2, weightOutput] = train_main(traindata,trainclass,validationdata,validationclass);

save('weights.mat','weightHidden1','weightHidden2', 'weightOutput');

%% Test with the validation set
[Predicted_digits_validation,Accuracy_validation] = mlp_test(validationdata,validationclass)

%% Test with the Training set
[Predicted_digits_training,Accuracy_training] = mlp_test(traindata,trainclass)

%% Test with all
[Predicted_digits_all,Accuracy_all] = mlp_test(data,class)
%% Test individual samples
load stroke_8_0094.mat
Predicted_digit = digit_classify(pos)

%% Train

function [best_weightHidden1,best_weightHidden2,best_weightOutput] = train_main(traindata,trainclass,validationdata,validationclass)

    maxEpochs = 200;
    
    % Initialisation
    hidden = 500; % number of the first hidden layer neurons
    hidden2 = 250; % number of the second hidden layer neurons
    J = zeros(1,maxEpochs); % loss function value vector initialisation
    rho = 0.00025; % learning rate
    eps = 1e-2;
    bias = 0;

    % Initialize weights
    weightHidden1 = (rand(910, hidden)-0.5) / 10;
    weightHidden2 = (rand(hidden+1, hidden2)-0.5) / 10;
    weightOutput = (rand(hidden2+1, 10)-0.5) / 10;

    % Initialize inputs and outputs
    inputv = cell(10,80);
    outputv = cell(10,80);

    % Initialize inputs

    for i = 1:size(traindata,2)
        for j = 1:size(traindata,1)

            % get data from cell
            n_traindata = cell2mat(traindata(j,i));
            n_traindata = n_traindata.pos;

            % Enhance and extract features
            C = feature_enhancer(n_traindata);
            C1 = feature_adder(C);
            C = feature_extractor(C);

            % Create input vectors with bias and store them to a cell array
            input = [C1; C; bias];
            inputv{j,i} = input;

            % Hot one encode class
            trainOutput = zeros(10, size(input, 2));
            n_trainOutput = trainclass(j,i);
            for a = 1:size(input, 2)
                trainOutput(n_trainOutput+1, a) = 1;
            end
            % Store hot one encoded classes to a cell array
            outputv{j,i} = trainOutput;
        end
    end

    % Train classifier
    t = 0;
    x = 1;
    best_accuracy = 0;
    while x == 1 % Train until stop criteria is met
        t = t+1;
    
        for i = 1:size(traindata,2)
            for j = 1:size(traindata,1)
                % Get the input data from the cell
                input = cell2mat(inputv(j,i));
                
                % Get the class form the cell
                trainOutput = cell2mat(outputv(j,i));
            
                % Feed-forward operation
                % Feed forward values to first hidden layer, use activation function relu and add bias
                Hidden1 = weightHidden1'*input;
                Hidden1 = reLu(Hidden1);
                Hidden1 = [Hidden1; bias];
        
                % Feed forward values to second hidden layer, use activation function relu and add bias
                Hidden2 = weightHidden2'*Hidden1;
                Hidden2 = reLu(Hidden2);
                Hidden2 = [Hidden2; bias]; 
        
                % Feed forward values to output layer, use activation softmax
                Output = weightOutput'*Hidden2;
                Output = softmax(Output);
        
                % Calculate loss with multiclass cross-entropy loss function
                J(t) = J(t) + sum(-((trainOutput.*(log(Output)))+((1-trainOutput).*log(1-Output)))); 
                
       
                % Backprogation
                % Computing sensitivities backwards in the network
        
                deltaOutput = Output - trainOutput;
                deltaHidden2 = (weightOutput(1:end-1, :) * deltaOutput) .*relu_d((Hidden2(1:end-1, :)));
                deltaHidden1 = (weightHidden2(1:end-1, :) * deltaHidden2) .*relu_d((Hidden1(1:end-1, :)));
        
                deltaWeightOutput = -rho * Hidden2 * deltaOutput';
                deltaWeightHidden2 = -rho * Hidden1* deltaHidden2';
                deltaWeightHidden1 = -rho * input* deltaHidden1';
                
                % Update weights
                weightOutput = weightOutput + deltaWeightOutput;
                weightHidden1 = weightHidden1 + deltaWeightHidden1;
                weightHidden2 = weightHidden2 + deltaWeightHidden2;
            end
        end
        % Cumulative loss
        J(t)
        
        % Calculate accuracy after every Epoch
        [~,Accuracy_validation] = mlp_test_2(validationdata,validationclass,weightHidden1,weightHidden2,weightOutput);
        if Accuracy_validation > best_accuracy
            best_accuracy = Accuracy_validation;
            best_weightHidden1 = weightHidden1;
            best_weightHidden2 = weightHidden2;
            best_weightOutput = weightOutput;
        end

        % Check stopping criteria
        if J(t) < eps % error very small
            disp("Learning good enough")
            disp("At")
            J(t)
            disp(J(1))
            x = 0;
            break;
        end

        if t == maxEpochs % max number of iterations reached
            disp("Max epochs reached")
            disp(J(1))
            x = 0;
            break;
        end

        if t > 1 % this is not the first epoch
            if norm(J(t) - J(t-1)) < 1e-10 % error changed very little
                disp("Improvement small enough")
                disp(J(1))
                x = 0;
                break;
            end
        end
        
    end
end



