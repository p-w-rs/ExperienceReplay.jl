using Test
using ExperienceReplay
using StatsBase

@testset "ExperienceReplay.jl" begin
    @testset "FlatBuffer Creation" begin
        # Test flat state space
        buffer = Buffer(4, 2, 1000)
        @test buffer isa FlatBuffer
        @test size(buffer.s) == (4, 1000)
        @test size(buffer.a) == (2, 1000)
        @test size(buffer.r) == (1, 1000)
        @test size(buffer.s′) == (4, 1000)
        @test size(buffer.t) == (1, 1000)
        @test length(buffer.ps) == 1000
        @test buffer.idx == 1
        @test buffer.len == 0
        @test buffer.maxl == 1000

        # Test tuple state space
        buffer = Buffer((4,), 2, 1000)
        @test buffer isa FlatBuffer
        @test size(buffer.s) == (4, 1000)
    end

    @testset "ImageBuffer Creation" begin
        # Test 2D state space
        buffer = Buffer((28, 28), 2, 1000)
        @test buffer isa ImageBuffer
        @test size(buffer.s) == (28, 28, 1, 1000)
        @test size(buffer.a) == (2, 1000)
        @test size(buffer.r) == (1, 1000)
        @test size(buffer.s′) == (28, 28, 1, 1000)

        # Test 3D state space
        buffer = Buffer((28, 28, 3), 2, 1000)
        @test buffer isa ImageBuffer
        @test size(buffer.s) == (28, 28, 3, 1000)
    end

    @testset "Store and Retrieve - FlatBuffer" begin
        buffer = Buffer(4, 2, 100)

        # Test storing single transition
        s = Float32[1, 2, 3, 4]
        a = 1
        r = 1.0f0
        s′ = Float32[2, 3, 4, 5]
        t = 0.0f0

        store!(buffer, s, a, r, s′, t)
        @test buffer.len == 1
        @test buffer.idx == 2
        @test buffer.s[:, 1] == s
        @test buffer.r[1, 1] == r
        @test buffer.s′[:, 1] == s′
        @test buffer.t[1, 1] == t

        # Test batch retrieval
        states, actions, rewards, next_states, terminals = get_batch!(buffer, 1)
        @test size(states) == (4, 1)
        @test size(actions) == (2, 1)
        @test size(rewards) == (1, 1)
        @test size(next_states) == (4, 1)
        @test size(terminals) == (1, 1)
    end

    @testset "Store and Retrieve - ImageBuffer" begin
        buffer = Buffer((2, 2), 2, 100)

        # Test storing single transition
        s = Float32[1 2; 3 4]
        a = 1
        r = 1.0f0
        s′ = Float32[2 3; 4 5]
        t = 0.0f0

        store!(buffer, s, a, r, s′, t)
        @test buffer.len == 1
        @test buffer.idx == 2
        @test buffer.s[:, :, 1, 1] == s
        @test buffer.r[1, 1] == r
        @test buffer.s′[:, :, 1, 1] == s′
        @test buffer.t[1, 1] == t
    end

    @testset "Priority Management" begin
        buffer = Buffer(4, 2, 100)

        # Store some transitions
        for i in 1:5
            store!(buffer, Float32[i,i,i,i], 1, Float32(i), Float32[i,i,i,i], 0.0f0)
        end

        # Test priority setting
        states, actions, rewards, next_states, terminals = get_batch!(buffer, 3)
        new_priorities = Float32[2.0, 3.0, 4.0]
        setp!(buffer, new_priorities)

        # Verify priorities were updated
        for (idx, priority) in zip(buffer.last_idxs, new_priorities)
            @test buffer.ps[idx] == priority
        end

        # Test priority reset
        resetp!(buffer, p=1.5f0)
        @test all(buffer.ps .== 1.5f0)
    end

    @testset "Circular Buffer Behavior" begin
        buffer = Buffer(4, 2, 3)  # Small buffer to test overflow

        # Store more transitions than buffer size
        for i in 1:5
            store!(buffer, Float32[i,i,i,i], 1, Float32(i), Float32[i,i,i,i], 0.0f0)
        end

        @test buffer.len == 3  # Buffer should be full
        @test buffer.idx == 3  # Should have wrapped around
    end

    @testset "Action Space Types" begin
        # Test discrete action space
        buffer_discrete = Buffer(4, 3, 100, discrete=true)
        store!(buffer_discrete, Float32[1,2,3,4], 2, 1.0f0, Float32[2,3,4,5], 0.0f0)
        @test sum(buffer_discrete.a[:, 1]) ≈ 1.0f0  # One-hot encoding

        # Test continuous action space
        buffer_continuous = Buffer(4, 2, 100, discrete=false)
        store!(buffer_continuous, Float32[1,2,3,4], Float32[0.5, -0.5], 1.0f0, Float32[2,3,4,5], 0.0f0)
        @test buffer_continuous.a[:, 1] == Float32[0.5, -0.5]
    end
end
