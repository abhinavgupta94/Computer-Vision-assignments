
local Convolution = nn.SpatialConvolution
local BatchNorm = nn.SpatialBatchNormalization
local ReLU = nn.ReLU
local Tanh = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear
local Dropout = nn.Dropout

local model = nn.Sequential()

model:add(Convolution(3, 100, 7, 7))
model:add(BatchNorm(100))
model:add(ReLU(true))
model:add(Dropout(0.4))
model:add(Max(2,2,2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(BatchNorm(150))
model:add(ReLU(true))
model:add(Dropout(0.4))
model:add(Max(2,2,2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(BatchNorm(250))
model:add(ReLU(true))
model:add(Dropout(0.4))
model:add(Max(2,2,2,2))
model:add(View(2250))
model:add(Linear(2250, 300))
model:add(ReLU(true))
model:add(Dropout(0.5))
model:add(Linear(300, 43))

-- model:cuda()
return model