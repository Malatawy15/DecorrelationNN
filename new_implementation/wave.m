function [accuracy, hidden_nodes] = wave()	
	input_patterns = dlmread('wave.data');
	input_patterns_size = size(input_patterns, 1);
	random_index = randperm(input_patterns_size);
	training_set = input_patterns(random_index(1 : 2700), :);
	testing_set = input_patterns(random_index(2701 : input_patterns_size), :);
  [w_hidden_input, sigma, w_output_hidden, error_rate] = backpropagation(40, 41, 3, training_set, 1, 0.1, 0.1, 4000);
  [hidden_nodes, accuracy] = classify(40, testing_set, w_hidden_input, sigma, w_output_hidden);
  fprintf('Number of hidden nodes = %d\n', size(sigma, 1));
end

