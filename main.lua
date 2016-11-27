require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'cunn'
require 'cudnn'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 1, 1--TODO: image size
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)


-- data transformation
function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

function transformInput(inp)
    local f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

-- get data sample and label
function getTrainSample(dataset, idx)

end

function getTrainLabel(dataset, idx)

end

--function getTestSample(dataset, idx)
--end

-- get iterator
function getIterator(dataset)

end
