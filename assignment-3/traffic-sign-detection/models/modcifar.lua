
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local ReLU = nn.ReLU
local Tanh = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model = nn.Sequential()

model:add(Convolution(3, 16, 5, 5))
model:add(BatchNorm(16))
model:add(ReLU(true))
model:add(Dropout(0.4))
model:add(Max(2,2,2,2))
model:add(Convolution(16, 128, 5, 5))
model:add(BatchNorm(128))
model:add(ReLU())
model:add(Dropout(0.4))
model:add(Max(2,2,2,2))
model:add(View(3200))
model:add(Linear(3200, 1000))
model:add(Tanh())
model:add(Dropout(0.5))
model:add(Linear(1000, 43))

model:cuda()
return model
