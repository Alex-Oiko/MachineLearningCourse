function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

size_of_input = size(z);
if(size_of_input(1:1) == 1)
    g = 1/(1+exp(-z));
else
    g = arrayfun(@(x) 1/(1+exp(-x)), z);
endif    


% =============================================================

end
