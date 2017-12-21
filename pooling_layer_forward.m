function [output] = pooling_layer_forward(input, layer)

%% function input:
% input.batch_size: batch_size of input
% input.height: height of input
% input.width : width of input
% input.data: the actual data of input
% input.data is of size (input.height*input.width*input.channel, input.batch_size)

% layer.k: kernel size of pooling operation
% layers.stride: stride of pooling operation
% layers.pad: pad of pooling operation


%% function output
% output: the output of inner_product_forward

% figure out the output shape
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
layer.pad = 0;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')

% set output shape
output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

% initialize output.data
output.data = zeros(h_out*w_out*c, batch_size);
modin=reshape(input.data,[h_in,w_in,c,batch_size]);
modout=reshape(output.data,[h_out,w_out,c,batch_size]);

for m = 1:h_out
    for n=1:w_out
        temp=modin((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,:,:);
        switch layer.act_type
            case 'MAX'
                modout(m,n,:,:)=max(max(temp));
            case 'AVE'
                modout(m,n,:,:)=mean(mean(temp));
        end
    end    % work out the average pooling and compute output.data     
end
output.data=reshape(modout,[h_out*w_out*c, batch_size]);
end

