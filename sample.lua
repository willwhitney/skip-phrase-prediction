require 'nn'
require 'optim'

require 'cutorch'
require 'cunn'

local SkipLSTM = require 'SkipLSTM'
local Loader = require 'Loader'

local network_name = arg[1]
local context_string = arg[2]
local temperature = arg[3]

local base_directory = 'networks'

function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, network_name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

local last_snapshot = getLastSnapshot(network_name)
local checkpoint = torch.load(path.join(base_directory, network_name, last_snapshot))


local model = checkpoint.model
opt = checkpoint.opt

-- we'll just use this for the vocab mappings
local loader = Loader(opt.datasetdir, 'train', opt.n_context, opt.n_skip, opt.n_predict)

local inputs = {}
for word in context_string:split(' ') do
    table.insert(inputs, loader.word_mappings[word])
end

local sequencer = model:findModules('nn.Sequencer')[1]
sequencer:remember()

local input = torch.Tensor(inputs)
print(input)

model:forward(input)

local outputs = {}

while #outputs < opt.n_predict do
    local output = model:forward(torch.zeros(1))
    output:div(temperature)
    local probs = torch.exp(output):squeeze()
    probs:div(torch.sum(probs))
    word = torch.multinomial(probs:float(), 1):resize(1):float()
    table.insert(outputs, word)
end

for _, word in outputs do
    print(loader.inverse_word_mappings[word])
end
