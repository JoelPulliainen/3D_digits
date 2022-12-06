function [traindata,trainclass, validationdata,validationclass] = data_splitter(data,class)
    rng("default")
    [m,n] = size(data) ;
    P = 0.80 ;
    idx = randperm(n)  ;
    traindata = data(:,idx(:,1:round(P*n))) ; 
    validationdata = data(:,idx(:,round(P*n)+1:end)) ;
    trainclass = class(:,idx(:,1:round(P*n)));
    validationclass = class(:,idx(:,round(P*n)+1:end));
    
    randRC = randperm(numel(traindata))
    traindata(:) = traindata(randRC)
    trainclass(:) = trainclass(randRC)

    randRT = randperm(numel(validationdata))
    validationdata(:) = validationdata(randRT)
    validationclass(:) = validationclass(randRT)
end