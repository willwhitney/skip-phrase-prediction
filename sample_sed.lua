require 'nn'
require 'optim'

require 'cutorch'
require 'cunn'

local SkipEncoderDecoder = require 'SkipEncoderDecoder'
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
-- local block1 = model.modules[1]

local sequencer = model:findModules('nn.Sequencer')[2]
sequencer:remember()

local encoder = model:findModules('nn.Sequential')[3]
local decoder = model.modules[#model.modules]
local word_embedder = model:findModules('nn.LookupTable')[1]

print(encoder)
print(decoder)

-- we'll just use this for the vocab mappings
local loader = Loader.create(opt.datasetdir, 'train', opt.n_context, opt.n_skip, opt.n_predict)

local inputs = {}
for _, word in ipairs(context_string:split(' ')) do
    table.insert(inputs, loader.word_mappings[word])
end

print("inputs:")
print(inputs)
local input = torch.Tensor(inputs):reshape(1, #inputs)
-- print(input)
local encoded = encoder:forward(input)
local outputs = {}

-- local output = decoder:forward{encoded}
-- table.insert(outputs, output)
while #outputs < opt.n_predict do
    -- print(outputs[#outputs])
    local current_input
    if #outputs > 0 then
        current_input = torch.Tensor{outputs[#outputs]}
        current_input = word_embedder:forward(current_input)
    else
        current_input = encoded
    end
    print(current_input)
    output = decoder:forward{current_input}
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
