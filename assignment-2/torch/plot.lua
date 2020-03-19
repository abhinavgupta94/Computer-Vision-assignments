require 'torch'
require 'image'

weights = torch.load("weights.th")
n = weights:reshape(10,28,28)

files = n[1]
for i=2,10 do
	files = torch.cat(files,n[i])
end

image.savePNG("weight.png", files)