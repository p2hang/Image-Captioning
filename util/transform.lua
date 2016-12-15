require 'image'
require 'torch'
local tnt = require 'torchnet'

local WIDTH, HEIGHT = 224, 224
local T = {}

-----------------------------------------------------------------------
---- Transform on the input and taget.
-----------------------------------------------------------------------

function resize(img)
    assert(img, "img is nil")
    return image.scale(img, WIDTH,HEIGHT):type('torch.DoubleTensor')
end

function greyToRGBIfNot(img)
    if(img:size()[1] ~= 3) then
        tmp = torch.DoubleTensor():resize(3, img:size()[2], img:size()[3])
        tmp[1]:copy(img[1])
        tmp[2]:copy(img[1])
        tmp[3]:copy(img[1])
        return tmp
    else
        return img
    end
end

function padInput(ip)
    local input = torch.IntTensor(32):fill(1)
    if ip:size()[1] < 32 then
        len = ip:size()[1]
    else
        len = 31
    end
    input[1] = 2
    for i = 1, len do
        input[i + 1] = ip[i]
    end
    return input
end

function padTarget(tg)
    local target = torch.IntTensor(32):fill(1)
    if tg:size()[1] < 32 then
        len = tg:size()[1]
    else
        len = 31
    end

    for i = 1, len do
        target[i] = tg[i]
    end
    target[len + 1] = 3
    return target
end




function T.onInputImage(inp)
    local f = tnt.transform.compose{
        [1] = resize,
        [2] = greyToRGBIfNot 
    }
    return f(inp)
end

function T.onInputText(inp)
    local f = tnt.transform.compose{
        [1] = padInput
    }
    return f(inp)
end

function T.onTarget(tg)
    local f = tnt.transform.compose{
        [1] = padTarget
    }
    return f(tg)
end

return T
