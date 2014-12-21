function [out_hidden, out_output] = forward_pass(input, w_hidden_input, sigma, w_output_hidden)
  input = [input' ; 1];
  net_hidden = w_hidden_input * input;
  out_hidden = sigmoid(net_hidden);

  for i=2 : size(out_hidden, 1)
    net_hidden(i) = net_hidden(i) + (sigma(i, :) * out_hidden);
    out_hidden(i) = sigmoid(net_hidden(i));
  end

  input_output = [out_hidden ; 1];
  net_output = w_output_hidden * input_output;
  out_output = sigmoid(net_output);
end

function out = sigmoid(mat)
  out = 1./(1.+(e.^(-mat)));
end
