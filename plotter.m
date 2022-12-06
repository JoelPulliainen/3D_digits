close all

[data,class] = data_collect();

% Plot figure
a = cell2mat(data(6,82));
C = a.pos;
function ed = plotter(C)

    % Plot a N x 3 dataset
    % Creates 2d and 3d plots of the original data
    % Creates also 2d and 3d plots of the enhanced data

    C1 = feature_enhancer(C);
    
    figure('name','original')
    size(C,1);
    x = C(:,1);
    y = C(:,2);
    hold on
    plot(x,y)
    plot(x,y,'r*')
    hold off
    
    figure('name','After feature enhancer')
    size(C1,1);
    x = C1(:,1);
    y = C1(:,2);
    hold on
    plot(x,y)
    plot(x,y,'r*')
    hold off
  
    
    figure('name','orgiginal 3d')
    size(C,1);
    x = C(:,1);
    y = C(:,2);
    z = C(:,3);
    hold on
    plot3(x,y,z,'r*')
    hold off
    
    figure('name','After feature enhancer 3d')
    size(C1,1);
    x = C1(:,1);
    y = C1(:,2);
    z = C1(:,3);
    hold on
    plot3(x,y,z,'r*')
    hold off

end