function plot_samples(data,title_str)
figure; 
plot(data(1,:),data(2,:),'.');
title(title_str); 
xlabel('x_1'); ylabel('x_2');