require 'torch'
require 'optim'
require 'os'
require 'xlua'require "util/load_data"
require 'cunn'
require 'cudnn'
require 'loadcaffe'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 224, 224--TODO: image size
-- local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

ImgCap = {}
include 'ImgCapModel/vggLSTM.lua'

torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)


-- data transformation
function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

function padInput(ip)
    local input = torch.IntTensor(32):fills(1)
    if ip:size()[1] < 32 then
        len = tg:size()[1]
    else
        len = 31
    end
    input[1] = 2
    for i = 1, len do 
        input[i + 1] = input[i]
    end
end


function transformInput(inp)
    local sample = {}
    local f = tnt.transform.compose{
        [1] = resize
    }
    local fc = tnt.transform.compose{
        [1] = padInput
    }
    sample.image = f(inp.image)
    sample.caption = fc(inp.caption)
    return sample 
end

function padTarget(tg)
    local target = torch.IntTensor(32):fills(1)
    if tg:size()[1] < 32 then
        len = tg:size()[1]
    else
        len = 31
    end

    for i = 1, len do 
        target[i] = tg[i]
    end
    target[len + 1] = 3
    return target 
end

function transformTarget(tg)
    local f = tnt.transform.compose{
        [1] = padTarget
    }
end




--function getTestSample(dataset, idx)
--end

-- get iterator
function getIterator(data_type)
    assert(data_type == "train" or data_type == "val")
    return tnt.ParallelDatasetIterator{
        nthread = 1,
        init = function() require 'torchnet' end,
        closure = function()
            
            local dataset = ld:loadData(data_type)
            return tnt.BatchDataset{
                batchsize = 1,
                dataset = tnt.ListDataset{
                    list = torch.range(1, dataset:size(1)):long(),
                    load = function(idx)
                        return{
                            input = transformInput({'image': ld:loadImage(data_type, dataset[idx].image_id),
                                     'caption': dataset[idx].caption}),
                            target = transformTarget(dataset[idx].caption)
                        }
                    end,
                }
            }
        end,
    }
end


local config = {} -- config for model, default as vgg
config.embeddingDim = opt.dim
config.imageOutputLayer = opt.imageLayer
config.imageModelPrototxt = 'models/caffe/' .. opt.protobuf or 'VGG_ILSVRC_19_layers_deploy.prototxt'
config.imageModelBinary = 'models/caffe/' .. opt.caffemodel or '/VGG_ILSVRC_19_layers.caffemodel'
config.num_words = opt.numWords

local model = require("models/" .. opt.model)
model.__init(config)



local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.Sequencer(nn.ClassNLLCriterion())
local clerr = tnt.ClassErrorMeter{topk = {3}}
local timer = tnt.TimeMeter()
local batch = 1 

if opt.cuda then 
    print("Using CUDA")
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(opt.gpuid)
    model:cuda()
    criterion:cuda()
    local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        engine.hooks.onSample = function(state)
        igpu:resize(state.sample.input:size() ):copy(state.sample.input)
        state.sample.input = igpu
        if state.sample.target then
            tgpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.target = tgpu
        end
    end
end


-- engine.hooks.onStart = function(state)
--     meter:reset()
--     clerr:reset()
--     timer:reset()
--     batch = 1
--     if state.training then
--         mode = 'Train'
--     else
--         mode = 'Val'
--     end
-- end





