module ExperienceReplay

export FlatBuffer, ImageBuffer, Buffer, store!, get_batch!, setp!, resetp!

using StatsBase: sample, weights

abstract type AbstractBuffer end

mutable struct FlatBuffer <: AbstractBuffer
    s::Matrix{Float32}  # state
    a::Matrix{Float32}  # action
    r::Matrix{Float32}  # reward
    s′::Matrix{Float32}  # next state
    t::Matrix{Float32}  # terminal
    ps::Vector{Float32}  # priorities
    last_idxs::Vector{Int64}
    idx::Int64
    len::Int64
    maxl::Int64
    actionf::Function
end

mutable struct ImageBuffer <: AbstractBuffer
    s::Array{Float32,4}  # state
    a::Matrix{Float32}  # action
    r::Matrix{Float32}  # reward
    s′::Array{Float32,4}  # next state
    t::Matrix{Float32}  # terminal
    ps::Vector{Float32}  # priorities
    last_idxs::Vector{Int64}
    idx::Int64
    len::Int64
    maxl::Int64
    actionf::Function
end

function Buffer(state_dim::Tuple{Int}, action_dim::Int, maxl::Int; discrete::Bool=true)
    return FlatBuffer(
        zeros(Float32, state_dim..., maxl),
        zeros(Float32, action_dim, maxl),
        zeros(Float32, 1, maxl),
        zeros(Float32, state_dim..., maxl),
        zeros(Float32, 1, maxl),
        ones(Float32, maxl),
        [], 1, 0, maxl, discrete ? x -> onehot(x, 1:action_dim) : x -> x
    )
end

function Buffer(state_dim::Tuple{Int,Int,Int}, action_dim::Int, maxl::Int; discrete::Bool=true)
    return ImageBuffer(
        zeros(Float32, state_dim..., maxl),
        zeros(Float32, action_dim, maxl),
        zeros(Float32, 1, maxl),
        zeros(Float32, state_dim..., maxl),
        zeros(Float32, 1, maxl),
        ones(Float32, maxl),
        [], 1, 0, maxl, discrete ? x -> onehot(x, 1:action_dim) : x -> x
    )
end

function Buffer(state_dim::Tuple{Int,Int}, action_dim::Int, maxl::Int; discrete::Bool=true)
    return ImageBuffer(
        zeros(Float32, state_dim..., 1, maxl),
        zeros(Float32, action_dim, maxl),
        zeros(Float32, 1, maxl),
        zeros(Float32, state_dim..., 1, maxl),
        zeros(Float32, 1, maxl),
        ones(Float32, maxl),
        [], 1, 0, maxl, discrete ? x -> onehot(x, 1:action_dim) : x -> x
    )
end

function store!(buffer::FlatBuffer, s, a, r, s′, t; p=1.0f0)
    buffer.s[:, buffer.idx] .= s
    buffer.a[:, buffer.idx] .= buffer.actionf(a)
    buffer.r[:, buffer.idx] .= r
    buffer.s′[:, buffer.idx] .= s′
    buffer.t[:, buffer.idx] .= t
    buffer.ps[buffer.idx] = p
    buffer.idx = buffer.idx + 1 > buffer.maxl ? 1 : buffer.idx + 1
    buffer.len = min(buffer.len + 1, buffer.maxl)
end

function store!(buffer::ImageBuffer, s, a, r, s′, t; p=1.0f0)
    buffer.s[:, :, :, buffer.idx] .= s
    buffer.a[:, buffer.idx] .= buffer.actionf(a)
    buffer.r[:, buffer.idx] .= r
    buffer.s′[:, :, :, buffer.idx] .= s′
    buffer.t[:, buffer.idx] .= t
    buffer.ps[buffer.idx] = p
    buffer.idx = buffer.idx + 1 > buffer.maxl ? 1 : buffer.idx + 1
    buffer.len = min(buffer.len + 1, buffer.maxl)
end

function get_batch!(buffer::FlatBuffer, n)
    idxs = 1:buffer.len
    idxs = sample(idxs, weights(1 ./ buffer.ps[idxs]), n; replace=false, ordered=false)
    buffer.last_idxs = idxs
    return buffer.s[:, idxs], buffer.a[:, idxs], buffer.r[:, idxs], buffer.s′[:, idxs], buffer.t[:, idxs]
end

function get_batch!(buffer::ImageBuffer, n)
    idxs = 1:buffer.len
    idxs = sample(idxs, pweights(buffer.ps[idxs]), n; replace=false, ordered=false)
    buffer.last_idxs = idxs
    return buffer.s[:, :, :, idxs], buffer.a[:, idxs], buffer.r[:, idxs], buffer.s′[:, :, :, idxs], buffer.t[:, idxs]
end

function setp!(buffer::AbstractBuffer, ps)
    buffer.ps[buffer.last_idxs] .= ps
end

function resetp!(buffer::AbstractBuffer; p=1.0f0)
    buffer.ps[:] .= p
end

# helper functions
function onehot(v, s)
    vector = zeros(Float32, size(s))
    vector[findfirst(isequal(v), s)] = 1.0f0
    return vector
end

end
