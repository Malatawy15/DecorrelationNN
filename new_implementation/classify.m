function [output, accuracy] = classify(input_size, test_set, w_hidden_input, sigma, w_output_hidden)
  input_set = test_set(:, 1:input_size);
  expected_output = test_set(:, input_size+1:end);
  output = [];
  accuracy = 0;
  test_set_size = size(test_set, 1);
  for i=1 : test_set_size 
    [output_hidden, network_output] = forward_pass(input_set(i, :), w_hidden_input, sigma, w_output_hidden);
    classification_result = hardlim(network_output);
    output = [output classification_result];
    accuracy = accuracy + (sum(classification_result' - expected_output(i, :)) == 0);
  end
  accuracy = accuracy * 100 / test_set_size;
end

function O = hardlim(value)
  O = value >= 0.5;
end
