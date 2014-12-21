function [w_hidden_input, sigma, w_output_hidden, stop] = prune(input_size, training_patterns, w_hidden_input, sigma, w_output_hidden)
  minimum = inf;
  min_index = -1;
  [hidden_outputs, output_outputs] = get_outputs(input_size, training_patterns, w_hidden_input, sigma, w_output_hidden);
  for i=1 : size(hidden_outputs, 1)
    node_correlation = 0;
    for j=1 : size(output_outputs, 1)
      outputs_hidden_i = hidden_outputs(i, :)';
      outputs_output_j = output_outputs(j, :)';
      covariance = cov([outputs_hidden_i  outputs_output_j])(2);
      variance_i = var(outputs_hidden_i);
      variance_j = var(outputs_output_j);
      node_correlation = node_correlation + (abs(covariance) / sqrt(variance_i * variance_j));
    end
    if (minimum > node_correlation)
      minimum = node_correlation;
      min_index = i;
    endif
  end
  if (minimum >= 0.5 || size(hidden_outputs, 1) == input_size)
    stop = 1;
  else
    w_hidden_input = [w_hidden_input(1 : min_index - 1, :) ; w_hidden_input(min_index + 1 : end, :)];
    w_output_hidden = [w_output_hidden(:, 1 : min_index - 1) w_output_hidden(:, min_index + 1 : end)];
    sigma = [sigma(1 : min_index - 1, :) ; sigma(min_index + 1 : end, :)];
    sigma = [sigma(:, 1 : min_index - 1) sigma(:, min_index + 1 : end)];
    stop = 0;
  endif
end

function [hidden_outputs, output_outputs] = get_outputs(input_size, training_patterns, w_hidden_input, sigma, w_output_hidden)
  input_set = training_patterns(:, 1:input_size);
  hidden_outputs = output_outputs = [];
  for i=1 : size(training_patterns, 1)
    [out_hidden, out_output] = forward_pass(input_set(i, :), w_hidden_input, sigma, w_output_hidden);
    hidden_outputs = [hidden_outputs out_hidden];
    output_outputs = [output_outputs out_output];
  end
end
