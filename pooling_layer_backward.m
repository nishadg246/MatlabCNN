function [input_od] = pooling_layer_backward(output, input, layer)

%% function input:
% input: input of pooling_layer_forward
% output: output of pooling_layer_forward

% layer.k: kernel size of pooling operation
% layer.stride: stride of pooling operation
% layer.pad: pad of pooling operation
stride=layer.stride;
k=layer.k;

%% function output
% input_od: gradient w.r.t input.data

% initialize
input_od = zeros(size(input.data));
modin=reshape(input.data,[input.height,input.width,input.channel,input.batch_size]);
modout=reshape(output.data,[output.height,output.width,output.channel,output.batch_size]);
moddiff=reshape(output.diff,[output.height,output.width,output.channel,output.batch_size]);
modinod=reshape(input_od,[input.height,input.width,input.channel,input.batch_size]);

switch layer.act_type
    case 'MAX'
        % work out the max pooling backward and compute input_od
        for b = 1:input.batch_size
            for ch = 1:input.channel
                for m = 1:output.height
                    for n=1:output.width
                        temp=modin((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,ch,b);
                        temp2 = (temp==modout(m,n,ch,b)).*moddiff(m,n,ch,b);
                        modinod((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,ch,b)=modinod((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,ch,b)+temp2;
                    end
                end
            end
        end
        input_od = reshape(modinod,size(input.data)); 
    case 'AVE'
        for b = 1:input.batch_size
            for ch = 1:input.channel
                for m = 1:output.height
                    for n=1:output.width
                        temp=ones(k,k);
                        temp2 = temp.*(moddiff(m,n,ch,b)/k/k);
                        modinod((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,ch,b)=modinod((m-1)*stride + 1 : (m-1)*stride + k, (n-1)*stride + 1 : (n-1)*stride + k,ch,b)+temp2;
                    end
                end
            end
        end
        input_od = reshape(modinod,size(input.data)); 
        % work out the ave pooling backward and compute input_od        

end

end