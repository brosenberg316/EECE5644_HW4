function net = init_mlp(num_perceptrons,data,init_weights)
% Initializes a MLP with given number of perceptrons
net = feedforwardnet(num_perceptrons);
net.performFcn = 'mse';
% net.input.processFcns = {'removeconstantrows','mapminmax'};
% Custom softplus activation function
net.layers{1}.transferFcn = 'softplus';

% Configure MLP dimensions to inputs and outputs
input_data = data(1,:);
output_data = data(2,:);
net = configure(net,input_data,output_data);

if nargin == 2
     % net.initFcn = 'initzero';
     net.b{1} = zeros(num_perceptrons,1);
     net.b{2} = zeros(1,1);
     net.IW{1} = xavier_init(num_perceptrons,1);
     net.LW{2,1} = xavier_init(1,num_perceptrons);
elseif nargin == 3
    net.b{1} = init_weights{1};
    net.b{2} = init_weights{2};
    net.IW{1} = init_weights{3};
    net.LW{2,1} = init_weights{4};
end
% Initialize all biases to zero
% net.b{1} = zeros(num_perceptrons,1);
% net.b{2} = zeros(1,1);
% net.b{1} = init_weights{1};
% net.b{2} = init_weights{2};
% net.IW{1} = init_weights{3};
% net.LW{2,1} = init_weights{4};
% net.initFcn = 'initzero';
% % Input input layer weight initialization using Xavier
% net.IW{1} = xavier_init(num_perceptrons,1);
% % Hidden layer weight initialization using Xavier
% net.LW{2,1} = xavier_init(1,num_perceptrons);
% Set max epochs to 200,000
net.trainParam.epochs = 200000;
% Use scaled conjugate gradient
net.trainFcn = 'trainscg';
net.trainParam.showWindow = false;