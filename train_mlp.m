clc
clear all
close 


%% Data collection and training set new values

[data,class] = data_collect();
[traindata,trainclass, validationdata,validationclass] = data_splitter(data,class);
[wHidden,wHidden2, wOutput] = train_main(traindata,trainclass);

save('weights.mat','wHidden','wHidden2', 'wOutput');

%% Test with the validation set
[Predicted_digits,Accuracy] = mlp_tester(validationdata,validationclass)

%% Test with  all
[Predicted_digits_all,Accuracy_all] = mlp_tester(data,class)
%% Test individual samples
load stroke_8_0094.mat
[confidence,Predicted_digit] = digit_classify(pos)

%% Train

function [pred_matrix,Accuracy_percentage] = mlp_tester(testdata,testclass)
    [m,n] = size(testdata);
    
    for i = 1:n
        for j = 1:m
            n_testdata = cell2mat(testdata(j,i));
            n_testdata = n_testdata.pos;
            [tmp,predclass] = digit_classify(n_testdata);
            realclass(j,i) = testclass(j,i);
            pred_matrix(j,i) = max(predclass);
        end
    end
    
    accuracy1 = realclass == pred_matrix;
    correct_classifications = sum(accuracy1,"all");
    incorrect = size(realclass,1)*size(realclass,2);
    Accuracy_percentage = correct_classifications/incorrect
end

function [wHidden,wHidden2,wOutput] = train_main(traindata,trainclass)

    maxEpochs = 1000;
    
    % Initialisation
    hidden = 500; % number of the first hidden layer neurons
    hidden2 = 250; % number of the second hidden layer neurons
    J = zeros(1,maxEpochs); % loss function value vector initialisation
    rho = 0.002; % learning rate
    eps = 1e-4;
    bias = 0.1;

    % Initialize weights
    wHidden = (rand(910, hidden)-0.5) / 10;
    wHidden2 = (rand(hidden+1, hidden2)-0.5) / 10;
    wOutput = (rand(hidden2+1, 10)-0.5) / 10;
    
    % Train classifier
    
    for i = 1:size(traindata,2)
        i % keep track of where we are while training
        for j = 1:size(traindata,1)

            j % keep track of where we are while training
            % get data from cell
            n_traindata = cell2mat(traindata(j,i));
            n_traindata = n_traindata.pos;

            % Enhance and extract features
            C = feature_enhancer(n_traindata);
            C1 = feature_adder(C);
            C = feature_extractorS(C);

            % Create input vector with bias
            extendedInput = [C1; C; bias];

            % Hot one encode class

            trainOutput = zeros(10, size(extendedInput, 2));
            n_trainOutput = trainclass(j,i)
            for a = 1:size(extendedInput, 2)
                trainOutput(n_trainOutput+1, a) = 1;
            end
            % Pass values to train the classifier
            [wHidden,wHidden2,wOutput] = mlp_train(extendedInput, trainOutput,maxEpochs,wHidden,wHidden2,wOutput,bias,eps,rho,J);
        end
    end

end

function [wHidden,wHidden2,wOutput] = mlp_train(extendedInput,trainOutput,maxEpochs,wHidden,wHidden2,wOutput,bias,eps,rho,J)
    t = 0;
    while 1 % iterative training "forever"
        t = t+1;
    
        % Feed-forward operation
        vHidden = wHidden'*extendedInput; % first hidden layer net activation
        yHidden = reLu(vHidden); % first hidden layer activation function
        yHidden = [yHidden; bias]; % first hidden layer extended output
    
        vHidden2 = wHidden2'*yHidden; % second hidden layer net activation
        yHidden2 = reLu(vHidden2); % second hidden layer activation function relu
        yHidden2 = [yHidden2; bias]; % second hidden layer extended output
   
        vOutput = wOutput'*yHidden2; % output layer net activation
        yOutput = softmax(vOutput); % output layer output activation softmax
        J(t) = sum(-((trainOutput.*(log(yOutput)))+((1-trainOutput).*log(1-yOutput)))); % Multiclass cross-entropy loss function
        
        if isnan(J(t))
            disp("NaN found on")
            
            return
        end
        if J(t) > 20
            disp("Anomaly")
            disp(J(1))
            break;
        end
        if J(t) < eps % the learning is good enough
            disp("Learning good enough")
            disp("At")
            J(t)
            disp(J(1))
            break;
        end
    
        if t == maxEpochs % too many epochs would be done
            disp("Max epochs reached")
            disp(J(1))
            break;
        end
    
        if t > 1 % this is not the first epoch
            if norm(J(t) - J(t-1)) < 1e-10 % the improvement is small enough
                disp("Improvement small enough")
                disp(J(1))
                break;
            end
        end

        % Backprogation
        % Update the sensitivities and the weights

        deltaOutput = yOutput - trainOutput;
        deltaHidden2 = (wOutput(1:end-1, :) * deltaOutput) .*relu_d((yHidden2(1:end-1, :)));
        deltaHidden1 = (wHidden2(1:end-1, :) * deltaHidden2) .*relu_d((yHidden(1:end-1, :)));

        deltawOutput = -rho * yHidden2 * deltaOutput';
        deltawHidden2 = -rho * yHidden* deltaHidden2';
        deltawHidden1 = -rho * extendedInput* deltaHidden1';
    
        wOutput = wOutput + deltawOutput;
        wHidden = wHidden + deltawHidden1;
        wHidden2 = wHidden2 + deltawHidden2;
    
    end
end



