function [out,final] = feature_extractor(as)

    % Init the matrix
    dimensions = zeros(30,30);

    % Normalize values
    C = normalize(as,'range');
    C = [C(:,1) C(:,2)];

    % Turn into matrix form
    for x = 1:size(C,1)
        for y = 1:size(C,2)
            eval(y) = C(x,y);
            eval(y) = round((eval(y)*29)+0.5,"TieBreaker","plusinf");
        end
        dimensions(eval(2),eval(1)) = dimensions(eval(2),eval(1))+1;
    end

    % Remove points with too few "hits"
    for x = 1:size(dimensions,1)
        for y = 1:size(dimensions,2)
            if dimensions(x,y) > 20
                final(x,y) = 1;
            else
                final(x,y) = 0;
            end
        end

    end
    % Flip to make it easier to inspect
    final = flip(final);

    % Flatten for the MLP
    out = reshape(final, [], 1);

end