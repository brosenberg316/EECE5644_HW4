function p_correct = plot_errors_svm(data,labels,svm,dataset_str)
y = svm.predict(data');
idx_correct = labels == y';
p_correct = sum(idx_correct)/length(labels);
figure() 
plot(data(1,idx_correct),data(2,idx_correct),'g.'); hold on,
plot(data(1,~idx_correct),data(2,~idx_correct),'r.'); axis equal,
Nx = 1001; Ny = 990; 
xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); 
dGrid = svm.predict([h(:),v(:)]); 
zGrid = reshape(dGrid,Ny,Nx);
contour(xGrid,yGrid,zGrid,1,'k--'); 
xlabel('x_1'), ylabel('x_2'); axis equal;
title({[dataset_str ' Predictions'],'Final Trained Gaussian SVM'}); 
legend('Correct Classification','Incorrect Classification','Decision Boundary');
hold off;