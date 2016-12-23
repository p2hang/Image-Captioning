require('nn')
require('nngraph')
require('rnn')
-- require 'loadcaffe'
require('paths')
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
    local imageModel
    if not paths.filep("models/vgg.t7") then
        require 'loadcaffe'
        imageModel = loadcaffe.load(config.imageModelPrototxt, config.imageModelBinary)
        for i = #imageModel.modules, config.imageOutputLayer + 1, -1 do
            imageModel:remove(i)
        end
        imageModel:clearState()
        -- torch.save("models/vgg.t7", imageModel)
    else
        imageModel = torch.load("models/vgg.t7")
    end

    self.imageModel = imageModel
    -- print(self.imageModel)

    -- visual rescale layer
    local imageOutputSize = (#self.imageModel.modules[config.imageOutputLayer].weight)[1]
    self.visualRescale = nn.Linear(imageOutputSize, config.embeddingDim)

    -- language model.
   LSTMcell = nn.Sequential()
           :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
           :add(nn.LSTM(config.embeddingDim, config.embeddingDim, 60))
           :add(nn.Linear(config.embeddingDim, self.num_words))
           -- :add(nn.LogSoftMax())
   self.LSTM = nn.Sequencer(LSTMcell)

    -- self.LSTM = nn.SeqLSTMP(config.embeddingDim, config.embeddingDim, self.num_words)
    -- self.LSTM.batchfirst = true
    collectgarbage()

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

    -- image nets

    self.outputImageModel = self.imageModel:forward(input.image)
    self.visualFeatureRescaled = self.visualRescale:forward(self.outputImageModel)

    -- vggLSTM -> LSTM -> sequencer -> recursor -> LSTMcell -> nn.LSTM, init h0 with visual feature
    self.LSTM.module.module.modules[1].userPrevOutput = self.visualFeatureRescaled

    -- LSTM
    self.text_embedding = self.embedding_vec:forward(input.text)
    self.text_embedding_transpose = self.embed_transpose:forward(self.text_embedding)

    self.LSTMout = self.LSTM:forward(self.text_embedding_transpose)
    self.output = self.output_transpose:forward(self.LSTMout)
    collectgarbage()
    return self.output
end

function vggLSTM:backward(input, grad)

    local gradT = self.output_transpose:backward(self.LSTMout, grad)


    local lstmInGrad = self.LSTM:backward(self.text_embedding_transpose, gradT)
    local lstmInGradT = self.embed_transpose:backward(self.text_embedding, lstmInGrad)
    -- backprop the rescale visual feature layer
    local gradVisualFeature = self.visualRescale:backward(self.outputImageModel, self.LSTM.module.module.modules[1].gradPrevOutput)

    self.embedding_vec:backward(input.text, lstmInGradT)
    collectgarbage()
end

function vggLSTM:predict(imageInput, beam_search, cuda)
    -- assert(imageInput:type() == "torch.DoubleTensor", "Type error, predict image input type should be torch.DoubleTensor")
    assert(imageInput:size()[1] == 3, "image channel error, predict image channel should be 3")
    assert(imageInput:size()[2] == 224, "Size error, predict image input size should be 224 * 224")
    assert(imageInput:size()[3] == 224, "Size error, predict image input size should be 224 * 224")

    -- self.LSTM.module.module.modules[1].userPrevCell = self.LSTM.module.module.modules[1].zeroTensor
    -- self.LSTM.module.module.modules[2].userPrevCell = self.LSTM.module.module.modules[2].zeroTensor
    -- self.LSTM.module.module.modules[2].userPrevOutput = self.LSTM.module.module.modules[2].zeroTensor

    local outputImageModel = self.imageModel:forward(imageInput)
    local visualFeatureRescaled = self.visualRescale:forward(outputImageModel)
    self.LSTM.module.module.modules[1].userPrevOutput = self.visualFeatureRescaled
    --Go id is 2, Eos id is 3
    local result = {}
    local textTensor
    local textInput
    local count = 1
    if not beam_search then 
        local lastWordIdx = 2

        while(lastWordIdx ~= 3) and count < 40 do 
            textTensor = torch.IntTensor({lastWordIdx})
            if cuda then 
                textTensor = textTensor:cuda()
            end
            if count ~= 1 then
                -- print("lllll_1")
                -- print("User prev output size")
                -- print(self.LSTM.module.module.modules[1].userPrevOutput:size())
                -- print("LSTM hidden size")
                -- print(self.prevHidden_2:size())
                self.LSTM.module.module.modules[1].userPrevOutput = self.prevHidden_1
                self.LSTM.module.module.modules[2].userPrevOutput= self.prevHidden_2
                -- print("lllll_2")
                self.LSTM.module.module.modules[1].userPrevCell = self.prevCell_1
                -- print("lllll_3")
                self.LSTM.module.module.modules[2].userPrevCell = self.prevCell_2
            end



            textInput = self.embedding_vec:forward(textTensor)
            -- print(textInput:size())
            -- if count == 1 then
            self.LSTMoutput = self.LSTM:forward(textInput)
            -- else
            --     print(self.prevHidden_2:size())
            --     self.LSTMoutput = self.LSTM:forward(self.prevHidden_2:view(1,-1))
            -- end

            self.prevCell_1 = self.LSTM.module.module.modules[1].cell
            self.prevHidden_1 = self.LSTM.module.module.modules[1].output

            self.prevCell_2 = self.LSTM.module.module.modules[2].cell
            self.prevHidden_2 = self.LSTM.module.module.modules[2].output

            val, idx = torch.max(self.LSTMoutput:float(),2)
            lastWordIdx = torch.totable(idx)[1][1] 
            print(lastWordIdx .. ' ' .. self.ivocab[lastWordIdx])
            table.insert(result,lastWordIdx)
            count = count + 1
        end
    end

    return self:convertWordIdxToSentence(result)
end 

function vggLSTM:convertWordIdxToSentence(tokenIndices)
    local sentence = {}
    for i = 1, #tokenIndices do 
        local word = self.ivocab[tokenIndices[i]]
        table.insert(sentence, word)
    end
    return table.concat(sentence, " ")
end


function vggLSTM:parameters()
    local modules = nn.Parallel()
    modules:add(self.embedding_vec)
           :add(self.visualRescale)
           :add(self.LSTM)
    return modules:parameters()
end

function vggLSTM:cuda()
    self.embedding_vec:cuda()

    self.embed_transpose:cuda()
    self.output_transpose:cuda()

    self.imageModel:cuda()
    self.visualRescale:cuda()

   self.LSTM:cuda()
end

return vggLSTM
