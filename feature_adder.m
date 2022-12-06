function out = feature_adder(C)

    % Find starting point
    start = [C(1,1);C(1,2);C(1,3)];

    % Find ending point
    epoint = [C(end,1);C(end,2);C(end,3)];

    % Calculate absolute X and Y distance between start and end
    x_y_dist = [abs(C(1,1)-C(end,1));abs(C(1,2)-C(end,2))];

    % Calculate 2d euclidian distance between start and end
    ed = sqrt((C(1,1) - C(end,1))^2+(C(1,2) - C(end,2))^2);
    
    out = [start;epoint;x_y_dist;ed];
end