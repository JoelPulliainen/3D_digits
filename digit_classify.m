function [confidence,predicted_class] = digit_classify(testdata)

    load weights.mat
    testdata = feature_enhancer2(testdata);
    testdata_stats = feature_adder(testdata);
    testdata = feature_extractor2d100(testdata);
    testdata = [testdata_stats;testdata];
    bias = 0.1;
    testdata = [testdata; bias];

    vHidden = wHidden'*testdata; % first hidden layer net activation
    yHidden = reLu(vHidden); % first hidden layer activation function relu
    yHidden = [yHidden; bias]; % first hidden layer extended output

    vHidden2 = wHidden2'*yHidden; % second hidden layer net activation
    yHidden2 = reLu(vHidden2); % second hidden layer activation function relu
    yHidden2 = [yHidden2; bias]; % second hidden layer extended output
   
    vOutput = wOutput'*yHidden2; % output layer net activation
    yOutput = soft(vOutput); % output layer softmax activation

    [confidence, predicted_class] = max(yOutput, [], 1);
    predicted_class = predicted_class-1; % -1 because indexing

end

