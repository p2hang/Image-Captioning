require 'torch'
require 'optim'
require 'os'
require 'xlua'
ld = require "util/load_data"

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 1, 1--TODO: image size
-- local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

ImgCap = {}
include 'models/LSTM.lua'

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
                            input = ld:loadImage(data_type, dataset[idx].image_id)
                            target = dataset[idx].caption
                        }
                    end,
                }
            }
        end,
    }
end

local model = require("models/" .. opt.model)
-- TODO:  load model param

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

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

