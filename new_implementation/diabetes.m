function [accuracy, hidden_nodes] = diabetes()	
	input_patterns = dlmread('diabetes.data');
	input_patterns_size = size(input_patterns, 1);
	random_index = randperm(input_patterns_size);
	training_set = input_patterns(random_index(1 : 384), :);
	testing_set = input_patterns(random_index(385 : input_patterns_size), :);
  [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(8, 10, 1, training_set, 1, 0.1, 0.1, 4000);
  [hidden_nodes, accuracy] = classify(8, testing_set, w_hidden_input, sigma, w_output_hidden);
  fprintf('Number of hidden nodes = %d\n', size(sigma, 1));
end

