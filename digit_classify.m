function C = digit_classify(testdata)

    % Classify digits written with LeapMotion sensor
    % Input N x 3 points
    % Output C digit
    % Adds data to the original to make it 1000x3 matrix
    % Calculate distinct features of digits
    % Create 30x30 matrix that represents the digit
    % Flatten the 30x30 matrix and feed it to a MLP with 2 hidden layers
    % Output a prediction of the digit


    % Get the weights of the neurons
    load weights.mat

    % Enhance, add and extract the features
    testdata = feature_enhancer(testdata);
    testdata_stats = feature_adder(testdata);
    testdata = feature_extractor(testdata);
    testdata = [testdata_stats;testdata];

    % Add bias
    bias = 0.1;
    testdata = [testdata; bias];

    % Classify
    % Feed forward values to first hidden layer, use activation function relu and add bias
    Hidden1 = weightHidden1'*testdata;
    Hidden1 = reLu(Hidden1);
    Hidden1 = [Hidden1; bias];

    % Feed forward values to second hidden layer, use activation function relu and add bias
    Hidden2 = weightHidden2'*Hidden1;
    Hidden2 = reLu(Hidden2);
    Hidden2 = [Hidden2; bias];

    % Feed forward values to output layer, use activation softmax
    Output = weightOutput'*Hidden2;
    Output = softmax(Output);

    [confidence, predicted_class] = max(Output, [], 1);
    C = predicted_class-1; % -1 because indexing

end

