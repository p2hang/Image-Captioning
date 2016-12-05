a = torch.load('data/processed/train.t7')

buckets = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50}
results = {}
count = 0
for idx = 1, #a do
	for idx2 = 1, #buckets do
		if a[idx].caption:size()[1] < buckets[idx2] then
			if not results[buckets[idx2]] then results[buckets[idx2]] = 0 end
			results[buckets[idx2]] = results[buckets[idx2]] + 1
			count = count + 1
			break
		end
	end
end

print(results)
print(#a - count)