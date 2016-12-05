require('nn')
require('nngraph')
require('rnn')
require 'loadcaffe'
-- require "util/load_data"
local vggLSTM, parent = torch.class('ImgCap.vggLSTM', 'nn.Module')

function vggLSTM:__init(config)
    self.vocab = ld:loadVocab()
    self.ivocab = ld:loadiVocab()
    self.num_words = #self.ivocab
    self.embedding_vec = nn.LookupTable(self.num_words, config.embeddingDim)
    self.embedding_vec.weight:copy(ld:loadVocab2Emb())

    -- Load image model after remove the last few layers.
    -- loadcaffe [https://github.com/szagoruyko/loadcaffe]
    -- And remove the layers after last linear layer. For vgg last is 42, (4096 -> 4096)
    local imageModel = loadcaffe.load(config.imageModelPrototxt, config.imageModelBinary)
    for i = #imageModel.modules, config.imageOutputLayer + 1, -1 do
        imageModel:remove(i)
    end

    self.imageModel = imageModel


    -- visual rescale layer
    local imageOutputSize = (#self.imageModel.modules[config.imageOutputLayer].weight)[1]
    self.visualRescale = nn.Linear(imageOutputSize, config.embeddingDim)

    -- language model.
    LSTMcell = nn.Sequential()
            :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
            :add(nn.Linear(config.embeddingDim, self.num_words))
    self.LSTM = nn.Sequencer(LSTMcell)
end

function vggLSTM:separateBatch(input)
    local batch_size = #input
    image_batch = {}
    caption_batch = {}
    for i = 1,batch_size do
        table.insert(image_batch, input[i].image)
        table.insert(caption_batch, input[i].caption)
    end
    return {['image']=image_batch, ['caption']=caption_batch}
end

function vggLSTM:forward(input)
    -- local separatedBatch = self:separateBatch(input)
    -- local inputImage = separatedBatch.image
    -- local inputText = separatedBatch.captions
    local inputImage = input[1].image -- To change
    local inputText = input[1].caption -- to change
    -- image nets
    self.outputImageModel = self.imageModel:forward(inputImage)
    self.visualFeatureRescaled = self.visualRescale:forward(self.outputImageModel)

    -- vggLSTM -> LSTM -> sequencer -> recursor -> LSTMcell -> nn.LSTM, init h0 with visual feature
    self.LSTM.module.module.modules[1].userPrevOutput = self.visualFeatureRescaled

    -- LSTM
    self.text_embedding = self.embedding_vec:forward(inputText)
    self.output = self.LSTM:forward(self.text_embedding)
    return self.output
end

function vggLSTM:backward(input, grad)
    local inputImage = input[1].image -- to change
    local inputText = input[1].caption -- to change
    -- backprop the language model
    local gradLSTMInput = self.LSTM:backward(self.text_embedding, grad)
    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, gradLSTMInput[1])

end




function vggLSTM:parameters()
    local modules = nn.Parallel()
    modules:add(self.embedding_vec)
           :add(self.imageModel)
           :add(self.LSTM)
    return modules:parameters()
end

return vggLSTM