function [accuracy, hidden_nodes] = ionosphere()	
	input_patterns = dlmread('ionosphere.data');
	input_patterns_size = size(input_patterns, 1);
	random_index = randperm(input_patterns_size);
	training_set = input_patterns(random_index(1 : 176), :);
	testing_set = input_patterns(random_index(177 : input_patterns_size), :);
  [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(34, 35, 1, training_set, 1, 0.1, 0.005, 4000);
  [hidden_nodes, accuracy] = classify(34, testing_set, w_hidden_input, sigma, w_output_hidden);
  fprintf('Number of hidden nodes = %d\n', size(sigma, 1));
end

