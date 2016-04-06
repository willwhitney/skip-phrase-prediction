require 'nn'
require 'optim'

-- require 'cutorch'
-- require 'cunn'

local SkipLSTM = require 'SkipLSTM'
local Loader = require 'Loader'

local network_name = arg[1]
local context_string = arg[2]
local temperature = arg[3]

local base_directory = 'networks'

function getLastSnapshot(name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

local last_snapshot = getLastSnapshot(network_name)
local checkpoint = torch.load(path.join(base_directory, network_name, last_snapshot))
opt = checkpoint.opt

local model = checkpoint.model
local block1 = model.modules[1]

local sequencer = block1:findModules('nn.Sequencer')[1]
sequencer:remember()

local block2 = model.modules[2]:clone()
table.remove(block2.modules, 1)


-- we'll just use this for the vocab mappings
local loader = Loader.create(opt.datasetdir, 'train', opt.n_context, opt.n_skip, opt.n_predict)

local inputs = {}
for _, word in ipairs(context_string:split(' ')) do
    table.insert(inputs, loader.word_mappings[word])
end

-- local input = torch.Tensor(inputs)
print("inputs:")
print(inputs)

for i = 1, #inputs do
    block1:forward(torch.Tensor{inputs[i]}:reshape(1,1))
end

local outputs = {loader.word_mappings["<PREDICT>"]}
-- print(outputs)

local assembled_model = nn.Sequential()
assembled_model:add(block1)
assembled_model:add(block2)
-- local output = assembled_model:forward(torch.zeros(1))
-- table.insert(outputs, output)
while #outputs < opt.n_predict do
    -- print(outputs[#outputs])
    local current_input = torch.Tensor{outputs[#outputs]}:reshape(1,1)
    -- print(current_input)
    output = assembled_model:forward(current_input)
    output:div(temperature)
    local probs = torch.exp(output):squeeze()
    probs:div(torch.sum(probs))
    word = torch.multinomial(probs:float(), 1):resize(1):float()[1]
    table.insert(outputs, word)
end

table.remove(outputs, 1)
for _, word in ipairs(outputs) do
    print(loader.inverse_word_mappings[word])
end
