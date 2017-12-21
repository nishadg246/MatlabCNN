function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

%% function input
% output.data: output data of inner_product_forward (layer.num x batch size)
% output.diff: the gradient w.r.t otuput.data   (layer.num x batch size)
%% function output
% param_grad.w: gradient w.r.t param.w
% param_grad.b: gradient w.r.t param.b
% input_od: gradient w.r.t input.data 

%% here begins inner product backward computation

% initialize the gradient w.r.t param
param_grad.w = zeros(size(param.w)); % gradient w.r.t param.w  % (input.height*input.width*input.channel, layer.num)
param_grad.b = zeros(size(param.b)); % gradient w.r.t param.b % (1, layer.num);
input_od = zeros(size(input.data)); % (input.height*input.width*input.channel, input.batch_size)

% modw=reshape(input.data,[input.height,input.width,input.channel,layer.num]);
% modin=reshape(input.data,[input.height,input.width,input.channel,input.batch_size]);
% modout=reshape(output.data,[output.height,output.width,output.channel,output.batch_size]);
% moddiff=reshape(output.diff,[output.height,output.width,output.channel,output.batch_size]);
% modinod=reshape(input_od,[input.height,input.width,input.channel,input.batch_size]);

input_od = param.w * output.diff;
param_grad.w = input.data * (output.diff.');
param_grad.b = sum(output.diff.'); 

    
    
% start to work here to compute param_grad.w, param_grad.b, input_od 


end
