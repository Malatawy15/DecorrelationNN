function [accuracy, hidden_nodes] = iris()	
	input_patterns = dlmread('iris.data');
	input_patterns_size = size(input_patterns, 1);
	random_index = randperm(input_patterns_size);
	training_set = input_patterns(random_index(1 : (input_patterns_size / 2)), :);
	testing_set = input_patterns(random_index((input_patterns_size / 2 + 1) : input_patterns_size), :);
  [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(4, 5, 3, training_set, 1, 0.1, 0.1, 4000);
  [hidden_nodes, accuracy] = classify(4, testing_set, w_hidden_input, sigma, w_output_hidden);
end

