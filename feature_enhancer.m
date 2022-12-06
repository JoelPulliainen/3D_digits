function out = feature_enhancer(C)
    C = normalize(C,'range');
    n = size(C,1);
    N = ceil(10000/n);
    C1 = C(1,:);
    for i = 1:n
        if i == n
            next_dot = C(i,:);
            x=linspace(old_dot(1),next_dot(1),N);
            y=linspace(old_dot(2),next_dot(2),N);
            z=linspace(old_dot(3),next_dot(3),N);
            C1 = [C1;new_dot];
            C1 = C1(1:10000,:,:);
            C = C1;
        else
            old_dot = C(i,:);
            next_dot = C(i+1,:);
            x=linspace(old_dot(1),next_dot(1),N);
            y=linspace(old_dot(2),next_dot(2),N);
            z=linspace(old_dot(3),next_dot(3),N);
            new_dot = [x',y',z'];
            C1 = [C1;new_dot];
        end
    end
    out = C;
end