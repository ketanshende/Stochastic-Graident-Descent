% Neural Network for XOR problem

clear all; close all; clc;

rng(2)
X = [0 0; 0 1; 1 0; 1 1]; % Inputs
y = [0, 1, 1, 0]; % XOR outputs
% y = [0, 1, 1, 1]; % OR outputs
% y = [1, 0, 0, 0]; % NOR outputs
% y = [0, 0, 0, 1]; % AND outputs
% y = [1, 1, 1, 0]; % NAND outputs
ninput = 2; % input nodes
nhidden = 2; % hidden nodes
nout = 1; % output nodes
W1 = rand(nhidden, ninput); % 2x2 weights for hidden neurons
W2 = rand(nout, nhidden); % 1x2 weights for output neurons
Bias1 = rand(nhidden,1);  % 2x1 bias values for hidden neurons
Bias2 = rand(nout,1);     % bias value for output neuron
rho = 0.1;               % learning rate
max_iter = 200000;         

for k=1:max_iter
    
    dW1 = 0;
    dW2 = 0;
    dBias1 = 0;
    dBias2 = 0;
    error = zeros(4,1);
    
    for j=1:4 % number of training example
        
        % Forward propagation
        
        z1 = W1*X(j,:)' + Bias1; % 2x1
        h1 = sigmoid(z1);        % 2x1
        z2 = W2*h1 + Bias2;      % 2x1
        h2 = sigmoid(z2);        % 1x1
        
        % Back propagation
        
        dz2 = h2 - y(j);        % error 1x1
        error(j) = dz2;
        dW2 = dW2 + dz2.*h1;    % weight change 2x1
        
        dz1 = (W2'*dz2).*sigmoid_derivative(h1); %2x1 
        dW1 = dW1 + dz1*X(j,:); % weight change 2x2
        dBias1 = dBias1 + dz1;  % bias 1 change 2x1
        dBias2 = dBias2 + dz2;  % bias 2 change 1x1
                 
        W1 = W1 - rho.*((dW1./4)); % new weights
        W2 = W2 - rho.*((dW2'./4)); % new weights
        Bias1 = Bias1 - rho.*(dBias1./4); % new bias values
        Bias2 = Bias2 - rho.*(dBias2./4); % new bias values
    
    end
    
    e(k) = abs(sum(error(:)));
    
end

% Test for correct output
input1 = [0 0; 0 1; 1 0; 1 1];
z1 = W1*input1' + Bias1;
h1 = sigmoid(z1);
z2 = W2*h1 + Bias2;
h2 = sigmoid(z2);
disp('Inputs (00 01 10 11):')
disp('Logic gate output:');
disp(h2)
plot(e,'LineWidth',3);
title('Stochastic Gradient Descent (XOR Neural Net)');
ylabel('Sum of Individual Errors');
xlabel('Iteration');

function y = sigmoid(x)

y = 1.0 ./ (1 + exp(-x));

end

function yd = sigmoid_derivative(x)

yd = x.*(1 - x);

end

