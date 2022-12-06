clc
clear all
close 


%% Data collection and training set new values

[data,class] = data_collect();
[traindata,trainclass, validationdata,validationclass] = data_splitter(data,class);

%% Train
[weightHidden1,weightHidden2, weightOutput] = train_main(traindata,trainclass);

save('weights.mat','weightHidden1','weightHidden2', 'weightOutput');

%% Test with the validation set
[Predicted_digits,Accuracy] = mlp_test(validationdata,validationclass)

%% Test with  all
[Predicted_digits_all,Accuracy_all] = mlp_test(data,class)
%% Test individual samples
load stroke_8_0094.mat
Predicted_digit = digit_classify(pos)

%% Train

function [weightHidden1,weightHidden2,weightOutput] = train_main(traindata,trainclass)

    maxEpochs = 2000;
    
    % Initialisation
    hidden = 500; % number of the first hidden layer neurons
    hidden2 = 250; % number of the second hidden layer neurons
    J = zeros(1,maxEpochs); % loss function value vector initialisation
    rho = 0.0025; % learning rate
    eps = 1e-4;
    bias = 0.1;

    % Initialize weights
    weightHidden1 = (rand(910, hidden)-0.5) / 10;
    weightHidden2 = (rand(hidden+1, hidden2)-0.5) / 10;
    weightOutput = (rand(hidden2+1, 10)-0.5) / 10;
    
    % Train classifier
    
    for i = 1:size(traindata,2)
        % i % keep track of where we are while training
        for j = 1:size(traindata,1)

            % j % keep track of where we are while training

            % get data from cell
            n_traindata = cell2mat(traindata(j,i));
            n_traindata = n_traindata.pos;

            % Enhance and extract features
            C = feature_enhancer(n_traindata);
            C1 = feature_adder(C);
            C = feature_extractor(C);

            % Create input vector with bias
            extendedInput = [C1; C; bias];

            % Hot one encode class

            trainOutput = zeros(10, size(extendedInput, 2));
            n_trainOutput = trainclass(j,i)
            for a = 1:size(extendedInput, 2)
                trainOutput(n_trainOutput+1, a) = 1;
            end
            % Pass values to train the classifier
            [weightHidden1,weightHidden2,weightOutput] = mlp_train(extendedInput, trainOutput,maxEpochs,weightHidden1,weightHidden2,weightOutput,bias,eps,rho,J);
        end
    end

end

function [weightHidden1,weightHidden2,weightOutput] = mlp_train(extendedInput,trainOutput,maxEpochs,weightHidden1,weightHidden2,weightOutput,bias,eps,rho,J)
    t = 0;
    while 1 % Train until stop criteria is met
        t = t+1;
    
        % Feed-forward operation
        % Feed forward values to first hidden layer, use activation function relu and add bias
        Hidden1 = weightHidden1'*extendedInput;
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
        J(t) = sum(-((trainOutput.*(log(Output)))+((1-trainOutput).*log(1-Output)))); 
        
        % Check stopping criteria
        if J(t) < eps % error very small
            disp("Learning good enough")
            disp("At")
            J(t)
            disp(J(1))
            break;
        end
    
        if t == maxEpochs % max number of iterations reached
            disp("Max epochs reached")
            disp(J(1))
            break;
        end
    
        if t > 1 % this is not the first epoch
            if norm(J(t) - J(t-1)) < 1e-10 % error changed very little
                disp("Improvement small enough")
                disp(J(1))
                break;
            end
        end

        % Backprogation
        % Computing sensitivities backwards in the network

        deltaOutput = Output - trainOutput;
        deltaHidden2 = (weightOutput(1:end-1, :) * deltaOutput) .*relu_d((Hidden2(1:end-1, :)));
        deltaHidden1 = (weightHidden2(1:end-1, :) * deltaHidden2) .*relu_d((Hidden1(1:end-1, :)));

        deltaWeightOutput = -rho * Hidden2 * deltaOutput';
        deltaWeightHidden2 = -rho * Hidden1* deltaHidden2';
        deltaWeightHidden1 = -rho * extendedInput* deltaHidden1';
        
        % Update weights
        weightOutput = weightOutput + deltaWeightOutput;
        weightHidden1 = weightHidden1 + deltaWeightHidden1;
        weightHidden2 = weightHidden2 + deltaWeightHidden2;
    
    end
end



