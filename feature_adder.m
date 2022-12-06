function out = feature_adder(C)
    start = [C(1,1);C(1,2);C(1,3)];
    epoint = [C(end,1);C(end,2);C(end,3)];
    x_y_dist = [abs(C(1,1)-C(end,1));abs(C(1,2)-C(end,2))];
    ed = sqrt((C(1,1) - C(end,1))^2+(C(1,2) - C(end,2))^2);
    out = [start;epoint;x_y_dist;ed];
end