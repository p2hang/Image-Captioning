require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'math'
ld = require "util/load_data"

local tnt = require 'torchnet'
local optParser = require 'opts'
local opt = optParser.parse(arg)
-- torch.setdefaulttensortype('torch.FloatTensor')


local train_set_size = 414113
local val_set_size = 201654
local min_loss = 100 -- err of the best model
local current_loss = 100 -- err of current bat

ImgCap = {}


torch.setdefaulttensortype('torch.DoubleTensor')

torch.manualSeed(opt.manualSeed)
require('ImgCapModel/' .. opt.model)

-----------------------------------------------------------------------
---- Get the dataset interator and transformation
-----------------------------------------------------------------------

-- get iterator
function getIterator(data_type)
    assert(data_type == "train" or data_type == "val")
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init = function() 
            require 'torchnet' 
            ld = require "util/load_data"
            end,
        closure = function()
            local dataset = ld:loadData(data_type)
            local image_features
            if data_type == "train" then
                image_features = torch.load('train_features.t7')
            elseif data_type == "val" then
                image_features = torch.load('val_features.t7')
            end
	    collectgarbage()

            local trans = require 'util/transform'
            tnt.BatchDataset.get = require 'util/get_in_batchdataset'

            return tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = tnt.ListDataset{
                    list = torch.range(1, #dataset):long(),
                    load = function(idx)
        		local id = string.format("%012d", dataset[idx].image_id)
                        return {
                            image = image_features[id],
                            text = trans.onInputText(dataset[idx].caption),
                            target = trans.onTarget(dataset[idx].caption)
                        }
                    end,
                }
            }
        end,
    }
end

-----------------------------------------------------------------------
---- Config or load the model.
-----------------------------------------------------------------------

 
local config = {} -- config for model, default as vgg
config.embeddingDim = opt.embeddingDim
config.imageOutputLayer = opt.imageLayer
config.imageModelPrototxt = 'models/caffe/' .. opt.protobuf or 'VGG_ILSVRC_19_layers_deploy.prototxt'
config.imageModelBinary = 'models/caffe/' .. opt.caffemodel or 'VGG_ILSVRC_19_layers.caffemodel'
config.num_words = opt.numWords
config.batchsize = opt.batchsize


local model

-- Init model from previous trained result.
if opt.cuda and opt.useWeights then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    print("Use pretrained weights for the vggLSTM.")
    model = torch.load(opt.weightsDir .. opt.model .. '_weights.t7')
    -- model:float()
    -- cudnn.convert(model, cudnn)
else
    require 'cunn'
    require 'cudnn'
    require 'cutorch' 
    model = ImgCap.vggLSTM(config)
--    local trained = torch.load(opt.weightsDir .. opt.model .. '_weights.t7')
--    model.embedding_vec = trained.embedding_vec
--    model.visualRescale = trained.visualRescale
--    model.LSTM = trained.LSTM 
--    trained = nil
    collectgarbage()   
end

-----------------------------------------------------------------------
---- Init variables and convert model to cuda if flag set.
-----------------------------------------------------------------------
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
local clerr = tnt.ClassErrorMeter{topk = {3}}
local timer = tnt.TimeMeter()
local torchTimer = torch.Timer()
local batch = 1
local lastFinishTime = -1
local startTime = -1

if opt.cuda then 
    print("Using CUDA")
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    -- cudnn.benchmark = true
    -- cudnn.fastest = true
    cutorch.setDevice(1)
    cudnn.convert(model, cudnn)

    model:cuda()
    criterion:cuda()

    engine.hooks.onSample = function(state)
        -- print(state.sample.input.image)
        state.sample.input.image = state.sample.input.image:cuda()
        state.sample.input.text = state.sample.input.text:cuda()
        if state.sample.target then
            state.sample.target = state.sample.target:cuda()
        end
    end

end

print(model)

-----------------------------------------------------------------------
---- Train with engine
-----------------------------------------------------------------------

engine.hooks.onStart = function(state)
    collectgarbage()
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    lastFinishTime = -1
    startTime = torchTimer:time().real
    if state.training then
        mode = 'Train'
        num_iters = math.floor(train_set_size / opt.batchsize)
    else
        mode = 'Val'
        num_iters = math.floor(val_set_size / opt.batchsize)
    end
end

--engine.hooks.onForward = function(state)
--end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)

    -- train timer and ETA
    local function timePerBatchAndETA()
        if lastFinishTime == -1 then
            lastFinishTime = startTime
        end
        local current = torchTimer:time().real
        local timeElapsed = current - lastFinishTime
        lastFinishTime = current
        local remainStep = num_iters - batch
        local step = (current - startTime) / batch
        local remainTimeTotal = remainStep * step

        return xlua.formatTime(timeElapsed), xlua.formatTime(remainTimeTotal)
    end

    if opt.verbose == true then
        local timeElapsed, ETA = timePerBatchAndETA()
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f, time: %s, ETA: %s",
            mode, batch, num_iters, meter:value()/opt.batchsize, timeElapsed, ETA))
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onBackward = function(state)
    local maxGrad = torch.max(state.gradParams) * opt.LR
    local minGrad = torch.min(state.gradParams) * opt.LR
    print("Range for the gradients: Max Gradients: " .. maxGrad .. " Min Gradients: " .. minGrad)
    --state.gradParams = torch.clamp(state.gradParams, -1, 1)
    collectgarbage()
end 


engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; time: %2.4f",
    mode, meter:value()/opt.batchsize,  timer:value()))
    -- print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    -- mode, meter:value(), clerr:value{k = 1}, timer:value()))
    collectgarbage()
    -- the error on end of the training
    current_loss = meter:value()/opt.batchsize
end

local lr = opt.LR 
local epoch = 1


if not opt.predictVal then
    print("Start Training!")
    logger = optim.Logger('logs/loss.log')
    logger:setNames{'Training Loss', 'Validation Loss'}
    while epoch <= opt.nEpochs do 
        
        local trainLoss
        local valLoss
        engine:train{
            network = model,
            criterion = criterion,
            iterator = getIterator('train'),
            optimMethod = optim.adam,
            maxepoch = 1,
            config = {
                learningRate = lr,
                momentum = opt.momentum,
--                wd = opt.weightDecay
            }
        }
        cutorch.synchronize()
        trainLoss = meter:value()/opt.batchsize

        
        engine:test {
            network = model,
            criterion = criterion,
            iterator = getIterator('val')
        }
        cutorch.synchronize()

        valLoss = meter:value()/opt.batchsize
        current_loss = valLoss

        logger:add{trainLoss, valLoss}

        -- save the model if it is the best so far
        if current_loss <= min_loss then
            print("update model")
            min_loss = current_loss
            cudnn.convert(model, nn)
            torch.save(opt.weightsDir .. opt.model .. '_weights.t7', model:clearState())
            cudnn.convert(model, cudnn)
        end

        print('Done with Epoch ' .. tostring(epoch))
        epoch = epoch + 1
    end 
else 
    print("Start Predicting")
    file = io.open("logs/prediction.log", "w")
    -- logger = optim.Logger('logs/prediction.log')
    -- logger:setNames{'Image ID', 'Caption'}
    local dataset = ld:loadData("val")
    local image_features = torch.load('val_features.t7')
    local trans = require 'util/transform'
    require 'ImgCapModel/vggLSTM'
    for i = 1, #dataset do 
    -- for i = 1, 100 do 
        local image = image_features[string.format("%012d", dataset[i].image_id)]
        if opt.cuda then 
            image = image:cuda()
        end
        -- predict(imageinput, beamsearch, cuda)
        local caption = model:predict(image, false, opt.cuda)
        model.clearState()
        -- logger:add{dataset[i].image_id, caption}
        file:write(dataset[i].image_id .. " " .. caption .. "\n")
        print("No. " .. i .. " sample finished")
    end 
end






print("The End!")

