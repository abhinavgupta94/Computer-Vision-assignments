local utils = {}

function utils.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

return utils
