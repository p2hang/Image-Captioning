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
    cmd:option('-batchsize',        56,             'Batch size for epochs')
    cmd:option('-nThreads',         3,              'Number of dataloading threads')
    cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.001,            'initial learning rate')
    cmd:option('-minimumLR',        0.001,          'minimum Learning Rate')
    cmd:option('-LRanneallingRate', 0.93,           'Annealing rate of learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-weightDecay',      1e-4,           'weight decay')
    cmd:option('-logDir',           'logs',         'log directory')
    cmd:option('-model',            'vggLSTM',      'Model to use for training')
    cmd:option('-verbose',          false,          'Print stats for every batch')
    cmd:option('-sub',              '',             'The submission file index')
    cmd:option('-AnnealingLR',      false,          'Whether we have annealing learning rate')
    cmd:option('-cuda',             false,          'Use cuda tensor')
    cmd:option('-gpuid',            1,              'gpuid')
    cmd:option('-nGPU',             1,              'number of GPU')
    cmd:option('-embeddingDim',     300,            'the embedding dimension of each vector')
    cmd:option('-imageLayer',       42,             'last layer of the image model')
    cmd:option('-numWords',         0,              'last layer of the image model')
    cmd:option('-protobuf',         "VGG_ILSVRC_19_layers_deploy.prototxt",    'protobuf to deploy the caffe model')
    cmd:option('-caffemodel',       "VGG_ILSVRC_19_layers.caffemodel",             'pretrained binary caffe model')
    cmd:option('-numWords',         38423,           'number of words in the dataset')
    cmd:option('-weightsDir',       'weights/',      'Path to save the model and weights')
    cmd:option('-useWeights',        false,              'use weights trained before')
    cmd:option('-predictVal',        false,           'Generate text with validaton set data')

    -- TODO: num words default value

    local opt = cmd:parse(arg or {})

    -- if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
    --     cmd:error('Invalid model ' .. opt.model)
    -- end

    return opt
end

return M

