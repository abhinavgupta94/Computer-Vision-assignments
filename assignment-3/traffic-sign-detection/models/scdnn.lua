local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Softmax = nn.SoftMax

local model  = nn.Sequential()

model:add(Convolution(3, 100, 7, 7))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(View(2250))
model:add(Linear(2250, 300))
model:add(Tanh())
model:add(Linear(300, 43))
model:add(Softmax())

return model