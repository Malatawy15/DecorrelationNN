function O = classify_new(w_hi, sigma, w_oh, input)
	x = [input' ; 1];
	net_hidden = (w_hi * x);
	out_hidden = (sigmoid(net_hidden));
	for j=2 : size(w_hi,1)
		net_hidden(j) = net_hidden(j) + (sigma(j, 1:end) * out_hidden);
		out_hidden(j) = sigmoid_element(net_hidden(j));
	end
	in_output = [out_hidden ; 1];
	net_output = (w_oh * in_output);
	out_output = (sigmoid(net_output));
	O = hardlim(out_output(1));
end

function h = hardlim(inp)
	if (inp>=0.5)
		h = 1;
	else 
		h = 0;
	endif
end

function o = sigmoid(number)
  o = [];
	for k=1 : size (number, 1)
		ox = [];
		for l=1 : size (number, 2)
			ox = [ox  1/(1+(e^(-number(k))))];
	 	end
		o = [o ; ox];
	end
end

function es = sigmoid_element(num)
  es = 1/(1+(e^(-num)));
end

