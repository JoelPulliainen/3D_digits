function [pred_matrix,Accuracy_percentage] = mlp_test(testdata,testclass)
    [m,n] = size(testdata);
    
    % Test the validation set
    for i = 1:n
        for j = 1:m
            n_testdata = cell2mat(testdata(j,i));
            n_testdata = n_testdata.pos;
            predclass = digit_classify(n_testdata);
            realclass(j,i) = testclass(j,i);
            pred_matrix(j,i) = max(predclass);
        end
    end

    % Calculate accuracy

    accuracy1 = realclass == pred_matrix;
    correct_classifications = sum(accuracy1,"all");
    incorrect = size(realclass,1)*size(realclass,2);
    Accuracy_percentage = correct_classifications/incorrect
end