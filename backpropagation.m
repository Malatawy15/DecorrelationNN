function [w_hidden_input, w_output_hidden] = backpropagation(input_size, hidden_size, output_size, training_patterns, learning_rate)

    %O = []
    
    w_hidden_input = ones(hidden_size, input_size + 1);
    theta_hidden = zeros(hidden_size, 1);
    
    w_output_hidden = ones(output_size, hidden_size + 1);
    theta_output = zeros(output_size, 1);
    
    input_patterns = training_patterns(1:size(training_patterns,1), 1:size(training_patterns,2)-1);
    target_output = training_patterns(1:size(training_patterns,1), end);
    
    for i=1 : size(training_patterns, 1)
        p = training_patterns(i,:);
        x = [input_patterns(i,:)' ; 1];
        net_hidden = (w_hidden_input * x);
        out_hidden = (sigmf(net_hidden, [1 0]));
        
        in_output = [out_hidden ; 1];
        net_output = (w_output_hidden * in_output);
        out_output = (sigmf(net_output, [1 0]));
        %O(end+1) = out_output
        
        delta_output = out_output .* (ones(size(out_output, 1),1) - out_output) .* (target_output(i,:) - out_output);
        delta_hidden = out_hidden .* (ones(size(out_hidden, 1),1) - out_hidden) .* (w_output_hidden(1:end, 1:end-1)' * delta_output);
        
        w_hi_increment = learning_rate .* (delta_hidden * x');
        w_oh_increment = learning_rate .* (delta_output * in_output');
        
        w_hidden_input = w_hidden_input + w_hi_increment;
        w_output_hidden = w_output_hidden + w_oh_increment;
        
    end
end