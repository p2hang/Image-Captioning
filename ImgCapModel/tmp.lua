a = torch.load('data/processed/train.t7')

buckets = {2,4,6,8,10,12,14,16,18,20}
results = {}
for idx = 1, #a do
	for idx2 = 1, #buckets do
		if a[idx].caption:size()[1] < buckets[idx2] then
			if not results[buckets[idx2]] then results[buckets[idx2]] = 0 end
			results[buckets[idx2]] = results[buckets[idx2]] + 1
			break
		end
	end
end

print(results)