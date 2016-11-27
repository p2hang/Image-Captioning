
data_loader = {}
data_loader.__index = data_loader

require 'paths'
local stringx = require 'pl.stringx'
function data_loader:init(opt)
	if not paths.filep("../data/processed/vocab.t7") then 
		self:buildVocab(opt)
	end

	if not paths.filep("../data/processed/train.t7") then
		self:buildData("train")
	end

	if not paths.filep("../data/processed/val.t7") then
		self:buildData("val")
	end

	if not paths.filep("../data/processed/initEmb.t7") then
		self:buildVocab2Emb(opt)
	end
end

function data_loader:loadVocab()
	return torch.load("../data/processed/vocab.t7")
end

function data_loader:loadiVocab()
	return torch.load("../data/processed/ivocab.t7")
end

function data_loader:buildVocab()
	print ("Building the vocab file")
	local _PAD = "_PAD"
	local _GO = "_GO"
	local _EOS = "_EOS"
	local _UNK = "_UNK"

	local vocab = {_PAD, _GO, _EOS, _UNK}
	local ivocab = {}
	for idx, item in pairs(vocab) do 
		ivocab[item] = idx 
	end

	local data_type = {"train", "val"}
	for _, dt in pairs(data_type) do 
		local filaname = "../data/processed".."_annotations.txt"
		for line in io.lines(filename) do 
			local divs = stringx.split(line, '\t')
			words = stringx.split(divs[2], ' ')
			for i = 1, #words fo 
				if vocab[words[i]] == nil then 
					vocab[words[i]] = #ivocab + 1
					ivocab[#ivocab + 1] = words[i]
				end
			end
		end
	end
	print("There are " .. #ivocab .. " in total.")
	torch.save("../data/processed/vocab.t7", vocab)
	torch.save("../data/processed/ivocab.t7", ivocab)
	print("Finish building the vocab")
end 

function data_loader:loadData(data_type)
	return torch.load("../data/processed/"..data_type..".t7")
end

function data_loader:buildData(config)
	local vocab = self:loadVocab()
	local ivocab = self:loadiVocab()
	local emb = torch.randn(#ivocab, config.embeddingDim)
	
end 

function data_loader:loadVocab2Emb()
	return torch.load("../data/processed/initEmb.t7")
end

function data_loader:buildVocab2Emb(opt)

end






return data_loader