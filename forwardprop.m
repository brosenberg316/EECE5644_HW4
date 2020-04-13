function da = forwardprop(dn,n,a,param)
%TANSIG.FORWARDPROP Forward propagate derivatives from input to output.

% Copyright 2012-2015 The MathWorks, Inc.

  d = log(1 + exp(a));
  da = bsxfun(@times,dn,d);
end