function [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(input_size, hidden_size, output_size, training_patterns, learning_rate_o, learning_rate_phi, desired_accuracy, max_iterations)
  w_hidden_input = rand(hidden_size, input_size + 1);
  w_output_hidden = rand(output_size, hidden_size + 1);
  sigma = tril(rand(hidden_size, hidden_size), -1);
  do
    [w_hidden_input, sigma, w_output_hidden, error_rate] = train(input_size, output_size, w_hidden_input, sigma, w_output_hidden, training_patterns, learning_rate_o, learning_rate_phi, desired_accuracy, max_iterations);
    [w_hidden_input, sigma, w_output_hidden, stop] = prune(input_size, training_patterns, w_hidden_input, sigma, w_output_hidden);
  until(stop == 1)
end
