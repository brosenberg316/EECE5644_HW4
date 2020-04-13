function plot_mse(mse,num_samples,epoch)
num_perceptrons = 1:length(mse);
figure();
bar(num_perceptrons,mse);
title(sprintf('10-fold Cross-Validation for %d Samples, Epoch %d',num_samples,epoch));
xlabel('Number of Perceptrons'); ylabel('MSE');
