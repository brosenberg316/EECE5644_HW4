function dn = backprop(da,n,a,param)
%TANSIG.BACKPROP Backpropagate derivatives from outputs to inputs

% Copyright 2012-2015 The MathWorks, Inc.

    d = 1./(1+exp(-a));
    dn = bsxfun(@times,da,d);
end
