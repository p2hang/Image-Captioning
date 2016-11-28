local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('The Microsoft COCO image captioning project')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',             '',             'Path to dataset')
    cmd:option('-val',              10,             'Percentage to use for validation set')
    cmd:option('-nEpochs',          30,             'Maximum epochs')
    cmd:option('-batchsize',        64,             'Batch size for epochs')
    cmd:option('-nThreads',         1,              'Number of dataloading threads')
    cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.1,            'initial learning rate')
    cmd:option('-minimumLR',        0.001,          'minimum Learning Rate')
    cmd:option('-LRanneallingRate', 0.93,           'Annealing rate of learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-weightDecay',      1e-4,           'weight decay')
    cmd:option('-logDir',           'logs',         'log directory')
    cmd:option('-model',            'init',         'Model to use for training')
    cmd:option('-verbose',          'false',        'Print stats for every batch')
    cmd:option('-continueTrain',    0,              'Continue to train on certain epoch')
    cmd:option('-sub',              '',             'The submission file index')
    cmd:option('-AnnealingLR',      false,          'Whether we have annealing learning rate')
    cmd:option('-protobuf',         '',             'protobuf to deploy the caffe model')
    cmd:option('-caffemodel',       '',             'pretrained binary caffe model')
    --[[
    -- Hint: Use this option to convert your code to use GPUs
    --]]
    cmd:option('-cuda',            false,             'Use cuda tensor')
    cmd:option('-gpuid',           1,                 'gpuid')
    cmd:option('-embeddingDim',   300,              'the embedding dimension of each vector')

    local opt = cmd:parse(arg or {})

    if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
        cmd:error('Invalid model ' .. opt.model)
    end

    return opt
end

return M
