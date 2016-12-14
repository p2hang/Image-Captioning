local argcheck = require 'argcheck'
return argcheck{
    {name='self', type='tnt.BatchDataset'},
    {name='idx', type='number'},
    call =
    function(self, idx)
        assert(idx >= 1 and idx <= self:size(), 'index out of bound')
        local samples = {}
        local maxidx = self.dataset:size()
        for i=1,self.batchsize do
            local idx = (idx - 1)*self.batchsize + i
            if idx > maxidx then
                break
            end
            idx = self.perm(idx, maxidx)
            local sample = self.dataset:get(idx)
            -- padding the input when insert to table
            if self.filter(sample) then
                table.insert(samples, sample)
            end
        end

        -- transform to vgg lstm use
        samples = self.makebatch(samples)
        local batchSample = {input = {image = samples.image, text = samples.text}, target = samples.target}
        collectgarbage()
        return batchSample
    end
}
