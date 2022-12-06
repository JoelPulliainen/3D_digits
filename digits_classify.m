clc
clear all
close all
%% a
data = cell(10,100);
class = 0:9;
class = repmat(class',[1 100]);

sizeR = [1 61] ;
num1 = 30;
R = zeros(sizeR); 
ix = randperm(numel(R));
ix = ix(1:num1);
R(ix) = 1;
R = R';

for i = 0:9
    for j = 1:100
        if j < 10
            F = sprintf('stroke_%d_000%d.mat',i,j);
        elseif j >= 10 && j ~= 100
            F = sprintf('stroke_%d_00%d.mat',i,j);
        else
            F = sprintf('stroke_%d_0%d.mat',i,j);
        end
        data{i+1,j} = load(F);
        
    end
end


% a = cell2mat(data(10,19));
% C = a.pos;
% ds = kmeans_matrix(C);
% ds
% C1 = a.pos;
%[idx,C] = kmeans(a.pos,12);
% C = normalize(C,'range');
% CA = [mean(C(:,1));mean(C(:,2));mean(C(:,3))]
% CV = [var(C(:,1));var(C(:,2));var(C(:,3))]
% CF = [CA;CV]
%q = 1;
% 
% for j = 1:100
%     a = cell2mat(data(3,j));
%     C1 = a.pos;
%     %C1 = normalize(C1,'range');
% %     [rows, columns] = size(C1);
% %     lastRow = int32(floor(0.15 * rows));
% %     C1(1:lastRow,:) = [];
% %     C1(end-lastRow:end,:) = [];
%     %C1 =  feature_enhancer(C1);
%     %[out,dimension] = feature_extractor2d100(C1);
%     x = C1(:,1);
%     y = C1(:,2);
%     figure(q),hold on
%     plot3(C1(:,1),C1(:,2),C1(:,3),'r*')
% %     plot(x,y)
% %     plot(C1(:,1),C1(:,2),'r*')
%     hold off
%     q = q + 1;
% end

% C = normalize(C,'range');
% C1 =  feature_enhancer(C1);
% [out,dimension] = feature_extractor2d100(C1);
% size(C,1)
% x = C(:,1);
% y = C(:,2);
% z = C(:,3);
% figure(1)
% hold on
% plot(x,y)
% plot(C(:,1),C(:,2),'r*')
% plot(mean(C(:,1)),mean(C(:,2)),'b*')
% hold off
% a = 1;
% i = 1;
% while a > 0
%     if i+1 > size(C,1)
%         a = 0;
%         break
%     else
%         old_dot = C(i,:);
%         next_dot = C(i+1,:);
%         dist_to_next = sqrt((old_dot(1) - next_dot(1))^2+(old_dot(2) - next_dot(2))^2+(old_dot(3) - next_dot(3))^2);
%         if dist_to_next > 0.1
%             new_dot = [((old_dot(1)+next_dot(1))/2) ((old_dot(2)+next_dot(2))/2) ((old_dot(3)+next_dot(3))/2)];
%             C = [C(1:i, :); new_dot; C(i+1:end, :)];
%         end
%         i = i+1;
%     end
% end
% size(C,1)
% figure(2)
% hold on
% plot(x,y)
% plot(C1(:,1),C1(:,2),'r*')
% % plot3(C(:,1),C(:,2),C(:,3),'r*')
% % plot3(mean(C(:,1)),mean(C(:,2)),mean(C(:,3)),'b*')
% hold off
% figure(3)
% % hold on
% plot3(C(:,1),C(:,2),C(:,3),'r*')

% hold off
% for i = 1:size(C,1)
%     for j = 1:size(C,1)
%         edist1(i,j) = sqrt((C(i,1) - C(j, 1))^2+(C(i,2) - C(j, 2))^2+(C(i,3) - C(j, 3))^2);
%     end
% end
% [v,~] = max(edist1,[], "all")
% for i = 1:size(C,1)
%     edist(i) = sqrt((CA(1) - C(i, 1))^2+(CA(2) - C(i, 2))^2+(CA(3) - C(i, 3))^2);
% end
% sa = mean(edist)
% va = var(edist)
%Split data to training set and test set
%%
[m,n] = size(data) ;
P = 0.70 ;
idx = randperm(n)  ;
traindata = data(:,idx(:,1:round(P*n))) ; 
testdata = data(:,idx(:,round(P*n)+1:end)) ;
trainclass = class(:,idx(:,1:round(P*n)));
testclass = class(:,idx(:,round(P*n)+1:end));
% 
% traindata = data;
% testdata = data;
% trainclass = class;
% testclass = class;

randRC = randperm(numel(traindata))
traindata(:) = traindata(randRC)
trainclass(:) = trainclass(randRC)

a = cell2mat(traindata(7,25));
C = a.pos

[out,dimension] = feature_extractor2d100(C)
C = normalize(C,'range');
x = C(:,1);
y = C(:,2);
figure(1)
hold on
plot(x,y)
plot(C(:,1),C(:,2),'r*')
plot(mean(C(:,1)),mean(C(:,2)),'b*')
hold off
% [train_idx, ~, test_idx] = dividerand(size(data_mat,1), 0.7, 0, 0.3);
% traindata = data_mat(train_idx,:);
% trainclass = labels(train_idx);
% testdata = data_mat(test_idx,:);
% testclass = labels(test_idx);

%% Train

% Template for implementing a shallow multilayer perceptron network

maxEpochs = 10000;

% Initialisation
hidden = 80; % number of hidden layer neurons
hidden2 = 35; % number of hidden layer neurons
hidden3 = 10; % number of hidden layer neurons
J = zeros(1,maxEpochs); % loss function value vector initialisation
rho = 0.001; % learning rate
eps = 1e-5;
bias = 1;
wHidden = (rand(401, hidden)-0.5) / 10;
wHidden2 = (rand(hidden+1, hidden2)-0.5) / 10;
wHidden3 = (rand(hidden2+1, hidden3)-0.5) / 10;
wOutput = (rand(hidden2+1, 10)-0.5) / 10;

% calculate accuracy


for i = 1:size(traindata,2)
    i
    for j = 1:size(traindata,1)
        j
        n_traindata = cell2mat(traindata(j,i));
        n_traindata = n_traindata.pos;
        n_traindata = normalize(n_traindata,1,'range');
%         [idx,C] = kmeans(n_traindata,18);
        %figure(j),plot(C(:,1),C(:,2),'r*')
%         C = normalize(C,'range');
%         C = [C(:,1);C(:,2)];
%         C1 = feature_enhancer2(n_traindata);
%         [C,Ce] = kmeans_matrix(n_traindata);
        C = feature_enhancer(n_traindata);
        C = feature_extractor2d100(C);
        extendedInput = [C; bias];
        trainOutput = zeros(10, size(extendedInput, 2));
        n_trainOutput = trainclass(j,i)
        for a = 1:size(extendedInput, 2)
            trainOutput(n_trainOutput+1, a) = 1;
        end
        [wHidden,wHidden2,wHidden3, wOutput] = mlp_train(extendedInput, trainOutput,maxEpochs,wHidden,wHidden2,wHidden3,wOutput,bias,eps,rho,J);
    end
end

%% test

randRCT = randperm(numel(testdata));
testdata(:) = testdata(randRCT);
testclass(:) = testclass(randRCT);
counts = zeros(1,10);
counts2 = zeros(1,10);

for i = 1:30
    for j = 1:10
        n_testdata = cell2mat(testdata(j,i));
        n_testdata = n_testdata.pos;
        %n_testdata = normalize(n_testdata,1,'range');
%         [idx,C] = kmeans(n_testdata,18);
%         C = normalize(C,'range');
%         plot3(C(:,1),C(:,2),C(:,3),'r*')
%         C1 = feature_enhancer2(n_testdata);
        C = feature_enhancer(n_testdata);
        C = feature_extractor2d100(C);
%         C = kmeans_matrix(C);
        extendedInput1 = [C; bias];
        [tmp,predclass] = mlp_test(extendedInput1,wHidden,wHidden2,wOutput,bias);
        realclass(j,i) = testclass(j,i);
        accuracy(j,i) = max(predclass)-1;
        counts(predclass) = counts(predclass)+1;
        counts2(predclass) = counts2(realclass(j,i)+1)+1;
    end
end

accuracy1 = realclass == accuracy;
correct_classifications = sum(accuracy1,"all");
incorrect = size(realclass,1)*size(realclass,2);
Accuracyper = correct_classifications/incorrect

%%
function out = feature_enhancer2(C)
    C = normalize(C,'range');
    %CV = [var(C(:,1));var(C(:,2));var(C(:,3))];
    ed_from_center_to_end = sqrt((0.5 - C(end,1))^2+(0.5 - C(end,2))^2+(0.5 - C(end,3))^2);
    ed_from_center_to_mid = sqrt((0.5 - C(round(end/2),1))^2+(0.5 - C(round(end/2),2))^2+(0.5 - C(round(end/2),3))^2);
    ed_from_center_to_first = sqrt((0.5 - C(1,1))^2+(0.5 - C(1,2))^2+(0.5 - C(1,3))^2);
    CM = [mean(C(:,1));mean(C(:,2));mean(C(:,3))];
    start = [C(1,1);C(1,2);C(1,3)];
    epoint = [C(end,1);C(end,2);C(end,3)];
    ed = sqrt((C(1,1) - C(end,1))^2+(C(1,2) - C(end,2))^2+(C(1,3) - C(end,3))^2);
    out = [start;epoint;ed;ed_from_center_to_end;ed_from_center_to_mid;ed_from_center_to_first;CM];
end
function out = feature_enhancer(C)
    C = normalize(C,'range');
    [rows, columns] = size(C);
    lastRow = int32(floor(0.1 * rows));
    C(1:lastRow,:) = [];
    C(end-lastRow:end,:) = [];
    a = 1;
    i = 1;
    b = 1;
    while a > 0
        if i+1 > size(C,1)
            if 4 > b
                i = 1;
                b = b+1;
            else
                a = 0;
                break
            end
            
        else
            old_dot = C(i,:);
            next_dot = C(i+1,:);
            dist_to_next = sqrt((old_dot(1) - next_dot(1))^2+(old_dot(2) - next_dot(2))^2+(old_dot(3) - next_dot(3))^2);
            if dist_to_next > 0.0001
                new_dot = [((old_dot(1)+next_dot(1))/2) ((old_dot(2)+next_dot(2))/2) ((old_dot(3)+next_dot(3))/2)];
                C = [C(1:i, :); new_dot; C(i+1:end, :)];
            end
            i = i+1;
        end
    end
    out = C;
end

function C = kmeans_matrix(C)
    %C = [C(:,1) C(:,2)];
    [~,C] = kmeans(C,4)
end

function out = feature_extractor(as)
    dimensions = zeros(20,20,20);
    C = normalize(as,'range');
    for x = 1:size(C,1)
        for y = 1:size(C,2)
            eval(y) = C(x,y);
            n_dim(y) = round((eval(y)*19)+0.5,"TieBreaker","plusinf");
        end
        dimensions(n_dim(1),n_dim(2),n_dim(3)) = 1;
    end
    dimensions = flip(dimensions);
    out = reshape(dimensions, [], 1);
end

function [out,dimensions] = feature_extractor2d(as)
    dimensions = zeros(10,10);
    C = normalize(as,'range');
    C = [C(:,1) C(:,2)];
    for x = 1:size(C,1)
        for y = 1:size(C,2)
            eval = C(x,y);
            if eval < 0.1
                n_dim(y) = 1;
            elseif eval >= 0.1 && eval < 0.2
                n_dim(y) = 2;
            elseif eval >= 0.2 && eval < 0.3
                n_dim(y) = 3;
            elseif eval >= 0.3 && eval < 0.4
                n_dim(y) = 4;
            elseif eval >= 0.4 && eval < 0.5
                n_dim(y) = 5;
            elseif eval >= 0.5 && eval < 0.6
                n_dim(y) = 6;
            elseif eval >= 0.6 && eval < 0.7
                n_dim(y) = 7;
            elseif eval >= 0.7 && eval < 0.8
                n_dim(y) = 8;
            elseif eval >= 0.8 && eval < 0.9
                n_dim(y) = 9;
            else
                n_dim(y) = 10;
            end

        end
        dimensions(n_dim(2),n_dim(1)) = 1;
    end
    dimensions = normalize(dimensions,'range');
    dimensions = flip(dimensions);
    out = reshape(dimensions, 100, 1);
end

function [out,dimensions] = feature_extractor2d100(as)
    dimensions = zeros(20,20);
    C = normalize(as,'range');
    C = [C(:,1) C(:,2)];
    for x = 1:size(C,1)
        for y = 1:size(C,2)
            eval(y) = C(x,y);
            eval(y) = round((eval(y)*19)+0.5,"TieBreaker","plusinf");
        end
        dimensions(eval(2),eval(1)) = 1;
    end
    dimensions = normalize(dimensions,'range');
    dimensions = flip(dimensions);
    out = reshape(dimensions, [], 1);
end

function y =  reLu(x)
    y=max(0,x);
end

function y = relu_d(x)
    y=heaviside(x);
end

function y =  lreLu(x)
    y=max(0,x) + (1e-2)*min(0,x);
end

function y =  reLu6(x)
    y = min(max(0,x),6);
end

function y = soft(x)
    ex=exp(x);
    y=ex/sum(ex);
end


function [wHidden,wHidden2,wHidden3,wOutput] = mlp_train(extendedInput,trainOutput,maxEpochs,wHidden,wHidden2,wHidden3,wOutput,bias,eps,rho,J)
    t = 0;
%     p = 0.7;
%     sizeR = [1 size(wHidden,2)] ;
%     num1 = size(wHidden,2)*p;
%     R1 = zeros(sizeR); 
%     ix = randperm(numel(R1));
%     ix = ix(1:num1);
%     R1(ix) = 1;
%     R1 = R1;
%     sizeR = [1 size(wHidden2,2)] ;
%     num1 = size(wHidden2,2)*p;
%     R2 = zeros(sizeR); 
%     ix = randperm(numel(R2));
%     ix = ix(1:num1);
%     R2(ix) = 1;
%     R2 = R2;
    while 1 % iterative training "forever"
        t = t+1;
    
        % Feed-forward operation
        vHidden = wHidden'*extendedInput; % hidden layer net activation
        %vHidden = vHidden.*R1';
        yHidden = reLu(vHidden); % hidden layer activation function
        yHidden = [yHidden; bias]; % hidden layer extended output
    
        vHidden2 = wHidden2'*yHidden; % hidden layer net activation
        %vHidden2 = vHidden2.*R2';
        yHidden2 = reLu(vHidden2); % hidden layer activation function
        yHidden2 = [yHidden2; bias]; % hidden layer extended output
    
%         vHidden3 = wHidden3'*yHidden2; % hidden layer net activation
%         yHidden3 = reLu(vHidden3); % hidden layer activation function
%         yHidden3 = [yHidden3; bias]; % hidden layer extended output
% %     
        vOutput = wOutput'*yHidden2; % output layer net activation
        yOutput = soft(vOutput); % output layer output without activation f

        J(t) = (max(-trainOutput.*(log(yOutput))));

        %J(t) = 0.5 * sum(sum((yOutput - trainOutput) .^ 2)); % error
        
        if isnan(J(t))
            disp("NaN found on")
            
            return
        end
        if J(t) > 15
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
                disp("Imporvement small enough")
                disp(J(1))
                break;
            end
        end

        % Backprogation
        % Update the sensitivities and the weights

        deltaOutput = yOutput - trainOutput;
%         deltaHidden3 = (wOutput(1:end-1, :) * deltaOutput) .*relu_d((yHidden3(1:end-1, :)));
%         deltaHidden2 = (wHidden3(1:end-1, :) * deltaHidden3) .*relu_d((yHidden2(1:end-1, :)));
        deltaHidden2 = (wOutput(1:end-1, :) * deltaOutput) .*relu_d((yHidden2(1:end-1, :)));
        deltaHidden1 = (wHidden2(1:end-1, :) * deltaHidden2) .*relu_d((yHidden(1:end-1, :)));

        deltawOutput = -rho * yHidden2 * deltaOutput';
%         deltawHidden3 = -rho * yHidden2* deltaHidden3';
        deltawHidden2 = -rho * yHidden* deltaHidden2';
        deltawHidden1 = -rho * extendedInput* deltaHidden1';
    
        wOutput = wOutput + deltawOutput;
        wHidden = wHidden + deltawHidden1;
        wHidden2 = wHidden2 + deltawHidden2;
%         wHidden3 = wHidden3 + deltawHidden3;
    
    end
end


function [tmp,testclass] = mlp_test(extendedInput,wHidden,wHidden2,wHidden3,wOutput,bias)
    % Testing with the test data
    
    
    vHidden = wHidden'*extendedInput; % hidden layer net activation
    yHidden = reLu(vHidden); % hidden layer activation function
    yHidden = [yHidden; bias]; % hidden layer extended output

    vHidden2 = wHidden2'*yHidden; % hidden layer net activation
    yHidden2 = reLu(vHidden2); % hidden layer activation function
    yHidden2 = [yHidden2; bias]; % hidden layer extended output
%     
%     vHidden3 = wHidden3'*yHidden2; % hidden layer net activation
%     yHidden3 = reLu(vHidden3); % hidden layer activation function
%     yHidden3 = [yHidden3; bias]; % hidden layer extended output
%     
    vOutput = wOutput'*yHidden2; % output layer net activation
    yOutput = soft(vOutput); % output layer output without activation f
    %disp(yOutput)
    [tmp, testclass] = max(yOutput, [], 1);

end