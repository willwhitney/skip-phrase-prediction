local data_loaders = {}

function data_loaders.load_atari_batch(id, mode)
    local data = torch.load(opt.datasetdir .. '/dataset_DQN_' .. opt.dataset_name .. '_trained/' .. mode .. '/images_batch_' .. id)

    local frame_interval = opt.frame_interval or 1

    if opt.gpu then
        data = data:cuda()
    end

    local inputs = {}
    local i = 1
    while i <= data:size(1) do
    -- while i <= 10 do
        -- stupid reshape so they still have a batch index
        table.insert(inputs, data[i]:reshape(1, data[i]:size(1), data[i]:size(2), data[i]:size(3)))
        i = i + frame_interval
    end

    return inputs
end

function data_loaders.load_random_atari_batch(mode)
    local id
    if mode == 'train' then
        id = math.random(opt.num_train_batches)
    elseif mode == 'test' then
        id = math.random(opt.num_train_batches)
    end
    return data_loaders.load_atari_batch(id, mode)
end


return data_loaders
