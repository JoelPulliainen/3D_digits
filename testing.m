close all

[data,class] = data_collect();

a = cell2mat(data(6,82));
C = a.pos;


for i = 1:size(data,2)
    for j = 1:size(data,1)
        a = cell2mat(data(j,i));
        C34 = a.pos;
        C34 = feature_enhancer(C34);
        startx(j,i) = C34(1,1);
        starty(j,i) = C34(1,2);
        startz(j,i) = C34(1,3);
        endx(j,i) = C34(end,1);
        endy(j,i) = C34(end,2);
        endz(j,i) = C34(end,3);
        CVx(j,i) = var(C34(:,1));
        CVy(j,i) = var(C34(:,2));
        CVz(j,i) = var(C34(:,3));
        ed(j,i) = feature_adder(C34);
    end
end


aAs = mean(ed,2);
mstartx = mean(startx,2);
mstarty = mean(starty,2);
mstartz = mean(startz,2);
mendx = mean(endx,2);
mendy = mean(endy,2);
mendz = mean(endz,2);
mvarx = mean(CVx,2);
mvary = mean(CVy,2);
mvarz = mean(CVz,2);
C2 = feature_enhancer(C);
C1 = feature_enhancer(C);


% C =  clean_data(C);
figure('name','org')
size(C,1);
x = C(:,1);
y = C(:,2);
hold on
plot(x,y)
plot(x,y,'r*')
hold off

figure('name','ppos')
size(C1,1);
x = C1(:,1);
y = C1(:,2);
hold on
plot(x,y)
plot(x,y,'r*')
hold off

[out,dimensions] = feature_extractor(C);
[out1,dimensions1] = feature_extractor(C1);
[out2,dimensions2] = feature_extractor(C2);

figure('name','dos')
size(C2,1);
x = C2(:,1);
y = C2(:,2);
hold on
plot(x,y)
plot(x,y,'r*')
hold off

figure('name','org3d')
size(C,1);
x = C(:,1);
y = C(:,2);
z = C(:,3);
hold on
plot3(x,y,z,'r*')
hold off

figure('name','ppos3d')
size(C1,1);
x = C1(:,1);
y = C1(:,2);
z = C1(:,3);
hold on
plot3(x,y,z,'r*')
hold off


figure('name','dos3d')
size(C2,1);
x = C2(:,1);
y = C2(:,2);
z = C2(:,3);
hold on
plot3(x,y,z,'r*')
hold off

function ed = feature_adder(C)
    start = [C(1,1);C(1,2);C(1,3)];
    epoint = [C(end,1);C(end,2);C(end,3)];
    x_y_dist = [abs(C(1,1)-C(end,1));abs(C(1,2)-C(end,2))];
    ed = sqrt((C(1,1) - C(end,1))^2+(C(1,2) - C(end,2))^2);
    out = [start;epoint;x_y_dist;ed];
end
