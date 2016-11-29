local nn = require 'nn'

local cnn = nn.Sequential()

cnn:add(nn.SpatialConvolutionMM(3,108,5,5,1,1,2))
cnn:add(nn.SpatialSubSampling(108, 2, 2, 2, 2))
cnn:add(nn.Tanh())
-- cnn:add(nn.SpatialContrastiveNormalization(12, torch.Tensor(4,4):fill(1)))

branch = nn.Concat(2)
branch_1 = nn.Sequential()
branch_1:add(nn.SpatialConvolutionMM(108,200,5,5,1,1,2))
branch_1:add(nn.SpatialSubSampling(200, 2, 2, 2, 2))
branch_1:add(nn.Tanh())

branch_2 = nn.Sequential()
branch_2:add(nn.SpatialSubSampling(108, 2, 2, 2, 2))

branch:add(branch_1)
branch:add(branch_2)

cnn:add(branch)

cnn:add(nn.Reshape((108+200)*8*8))
cnn:add(nn.Linear((108+200)*8*8, 100))
cnn:add(nn.Tanh())
cnn:add(nn.Linear(100, 100))
cnn:add(nn.Tanh())
cnn:add(nn.Linear(100, 43))

return cnn
