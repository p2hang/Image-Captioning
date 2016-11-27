
data_loader = {}
data_loader.__index = data_loader

require 'paths'
local stringx = require 'pl.stringx'
function data_loader:init(opt)
	self.embeddingDim = opt.embeddingDim

	if not paths.filep("data/processed/vocab.t7") then 
		self:buildVocab(opt)
	end

	if not paths.filep("data/processed/train.t7") then
		self:buildData("train")
	end

	if not paths.filep("data/processed/val.t7") then
		self:buildData("val")
	end

	if not paths.filep("data/processed/initEmb.t7") then
		self:buildVocab2Emb(opt)
	end
end

function data_loader:loadVocab()
	return torch.load("data/processed/vocab.t7")
end

function data_loader:loadiVocab()
	return torch.load("data/processed/ivocab.t7")
end

function data_loader:buildVocab()
	print ("Building the vocab file")
	local _PAD = "_PAD"
	local _GO = "_GO"
	local _EOS = "_EOS"
	local _UNK = "_UNK"

	local vocab = {}
	local ivocab = {_PAD, _GO, _EOS, _UNK}
	for idx, item in pairs(ivocab) do 
		vocab[item] = idx 
	end

	local data_type = {"train", "val"}
	for _, dt in pairs(data_type) do 
		local filename = "data/processed/".. dt .."_annotations.txt"
		if not paths.filep(filename) then
			error(filename .. " does not exist. Please do preprocess.")
		end
		for line in io.lines(filename) do 
			local divs = stringx.split(line, '\t')
			words = stringx.split(divs[2], ' ')
			for i = 1, #words do
				if vocab[words[i]] == nil then 
					vocab[words[i]] = #ivocab + 1
					ivocab[#ivocab + 1] = words[i]
				end
			end
		end
	end
	print("There are " .. #ivocab .. " in total.")
	torch.save("data/processed/vocab.t7", vocab)
	torch.save("data/processed/ivocab.t7", ivocab)
	print("Finish building the vocab")
end 

function data_loader:loadData(data_type)
	return torch.load("data/processed/"..data_type..".t7")
end

function data_loader:buildData(data_type)
	-- data_type refers to "train" or "val"
	print("Building " .. data_type .. " data")
	local vocab = self:loadVocab()
	data = {}
	for line in io.lines("data/processed/" .. data_type .. "_annotations.txt") do 
		local instance = {}
		local vals = stringx.split(line, '\t')
		instance.image_id = tonumber(vals[1])
		local caption = stringx.split(vals[2], ' ')
		ids = torch.IntTensor(#caption)
		for i = 1, #caption do 
			ids[i] = tonumber(vocab[caption[i]])
		end
		instance.caption = ids
		data[#data + 1] = instance
	end
	torch.save("data/processed/" .. data_type .. '.t7', data)
	print("Build " .. data_type .. " data complete")
end 

function data_loader:loadVocab2Emb()
	return torch.load("data/processed/initEmb.t7")
end

function data_loader:buildVocab2Emb(opt)
	print("Building Vocab to embedding with GloVe")
	local vocab = self:loadVocab()
	local ivocab = self:loadiVocab()
	local emb = torch.randn(#ivocab, opt.embeddingDim) * 0.05
	if not paths.filep('data/GloVe/glove.840B.300d.txt') then 
		error("glove.840B.300d.txt does not exist. Please download it.")
	end
	file = io.open('data/GloVe/glove.840B.300d.txt','r')
	local count = 0
	while true do 
		local line = file:read()

		if line == nil then break end 

		vals = stringx.split(line, ' ')
		if vocab[vals[1]] ~= nil then 
			for i = 2, #vals do 
				emb[vocab[vals[1]]][i - 1] = tonumber(vals[i])
			end
			count = count + 1
			if count == #ivocab then 
				break 
			end
		end 
	end

	print("There are " .. (#ivocab - count) .. " words do not appear in GloVe")
	torch.save("data/processed/initEmb.t7", emb)
	print("Build initial embedding complete")
end






return data_loader