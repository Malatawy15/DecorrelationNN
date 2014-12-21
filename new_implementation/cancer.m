function [accuracy, hidden_nodes] = cancer()	
	input_patterns = dlmread('cancer.data');
  input_patterns = input_patterns(:, 2 : end);
  input_patterns(:, end) = (input_patterns(:, end) .- 2) ./ 2;
	input_patterns_size = size(input_patterns, 1);
	random_index = randperm(input_patterns_size);
	training_set = input_patterns(random_index(1 : 350), :);
	testing_set = input_patterns(random_index(351 : input_patterns_size), :);
  [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(9, 12, 1, training_set, 1, 0.1, 0.03, 40000);
  [hidden_nodes, accuracy] = classify(9, testing_set, w_hidden_input, sigma, w_output_hidden);
  fprintf('Number of hidden nodes = %d\n', size(sigma, 1));
end

