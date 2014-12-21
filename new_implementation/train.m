function [w_hidden_input, sigma, w_output_hidden, error_rate] = train(input_size, output_size, w_hidden_input, sigma, w_output_hidden, training_patterns, learning_rate_o, learning_rate_phi, desired_accuracy, max_iterations)
  patterns_size = size(training_patterns, 1);
  error_rate = desired_accuracy + 1;
  counter = 0;
  while(error_rate > desired_accuracy && counter < max_iterations)
    training_patterns = training_patterns(randperm(patterns_size), :);
    input_set = training_patterns(:, 1:input_size);
    expected_output = training_patterns(:, input_size+1:end);
    
    error_rate = 0;

    for i=1 : patterns_size
      input = input_set(i, :); % row vector
      output = expected_output(i, :); % row vector
      
      [out_hidden, out_output] = forward_pass(input, w_hidden_input, sigma, w_output_hidden); % column vectors

      delta_output = out_output .* (1 .- out_output) .* (output' - out_output); % column vector (|o| x 1)
      delta_hidden = out_hidden .* (1 .- out_hidden) .* (w_output_hidden(:, 1 : end - 1)' * delta_output); % column vector (|h| x 1)
      
      error_rate = error_rate + (0.5 * sum((output' - out_output).^2));

      w_hi_increment_o = learning_rate_o .* (delta_hidden * [input, 1]);
      w_oh_increment_o = learning_rate_o .* (delta_output * [out_hidden', 1]);
      sigma_increment_o = learning_rate_o .* (delta_hidden * out_hidden');

      w_hi_increment_phi = -learning_rate_phi .* ((out_hidden .* out_hidden .* (1 .- out_hidden)) * [input, 1]);
      sigma_increment_phi = -learning_rate_phi .* ((out_hidden .* out_hidden .* (1 .- out_hidden)) * out_hidden');

      w_hidden_input = w_hidden_input + w_hi_increment_o + w_hi_increment_phi;
      w_output_hidden = w_output_hidden + w_oh_increment_o;
      sigma = tril(sigma + sigma_increment_o + sigma_increment_phi, -1);
    end

    error_rate = error_rate / patterns_size;
    counter = counter + 1;
  endwhile
end
