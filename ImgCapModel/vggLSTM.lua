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
    self.embed_transpose = nn.Transpose({1,2})
    self.output_transpose = nn.Transpose({1,2})

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
           :add(nn.SoftMax())
   self.LSTM = nn.Sequencer(LSTMcell)

    -- self.LSTM = nn.SeqLSTMP(config.embeddingDim, config.embeddingDim, self.num_words)
    -- self.LSTM.batchfirst = true

end

function vggLSTM:separateBatch(input)
    local batch_size = #input

    local image_size = input[1].image:size()
    local caption_size = input[1].caption:size()
    local image_tensor = torch.DoubleTensor(batch_size, image_size[1],image_size[2],image_size[3])
    local caption_tensor = torch.DoubleTensor(batch_size, caption_size[1])

    for i = 1, batch_size do 
        image_tensor[i]:copy(input[i].image)
        caption_tensor[i]:copy(input[i].caption)
    end
    -- print(image_tensor:size())
    -- print(caption_tensor:size())

    return {['image']=image_tensor, ['caption']=caption_tensor}
end

function vggLSTM:forward(input)
    local separatedBatch = self:separateBatch(input)
    local inputImage = separatedBatch.image
    local inputText = separatedBatch.caption
    -- local inputImage = input[1].image -- To change
    -- local inputText = input[1].caption -- to change
    -- image nets
    self.outputImageModel = self.imageModel:forward(inputImage)
    self.visualFeatureRescaled = self.visualRescale:forward(self.outputImageModel)

    -- vggLSTM -> LSTM -> sequencer -> recursor -> LSTMcell -> nn.LSTM, init h0 with visual feature
    print(self.LSTM.module.module.modules[1].userPrevOutput)
    print(self.visualFeatureRescaled:size())
    self.LSTM.module.module.modules[1].userPrevOutput = self.visualFeatureRescaled

    -- LSTM
    self.text_embedding = self.embedding_vec:forward(inputText)
    self.text_embedding_transpose = self.embed_transpose:forward(self.text_embedding)
    -- print(self.text_embedding_transpose:size())
    self.LSTMout = self.LSTM:forward(self.text_embedding_transpose)
    self.output = self.output_transpose:forward(self.LSTMout)

    -- print(self.output:size())
    return self.output
end

function vggLSTM:backward(input, grad)
    local separatedBatch = self:separateBatch(input)
    local inputImage = separatedBatch.image
    local inputText = separatedBatch.caption
    -- local inputImage = input[1].image -- to change
    -- local inputText = input[1].caption -- to change
    -- backprop the language model
    print(grad:size())
    local gradT = self.output_transpose:backward(self.LSTMout, grad)
    print(gradT:size())
    local lstmInGrad = self.LSTM:backward(self.text_embedding_transpose, gradT)
    local lstmInGradT = self.embed_transpose:backward(self.text_embedding, lstmInGrad)
    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, self.LSTM.gradPrevOutput)

    self.embedding_vec:backward(inputText, lstmInGradT)
end




function vggLSTM:parameters()
    local modules = nn.Parallel()
    modules:add(self.embedding_vec)
           :add(self.imageModel)
           :add(self.LSTM)
    return modules:parameters()
end

return vggLSTM