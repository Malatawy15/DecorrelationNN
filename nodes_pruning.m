function O = nodes_pruning(out_hidden, error_output, output_size, hidden_size) 
%error output will have to be computed in the function for every i,j, this is just for simplicity
%both out_hidden and error output are nx1 vectors, where elements in every row are from the different training patterns    
    sum = 0
    min = 10000000
    min_idx = 0
    for i = 1:hidden_size
        sum = 0
        for j = 1:output_size
            mat = cov([out_hidden error_output])
            covariance = pij(1,2)
            denom = sqrt(var(i) * var(j))
            pij = covariance / denom
            sum = sum + pij
        end
        avg = sum/output_size
        if avg < min
            min = avg
            minIdx = i
        end
    end
end