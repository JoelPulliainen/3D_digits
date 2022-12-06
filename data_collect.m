function [data,class] = data_collect()
    data = cell(10,100);
    class = 0:9;
    class = repmat(class',[1 100]);
    addpath('training_data')  

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
end