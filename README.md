# ExperienceReplay.jl

This module provides an implementation of a replay buffer, a fundamental component in off-policy reinforcement learning algorithms such as DQN and DDPG. The replay buffer stores transitions (`state`, `action`, `reward`, `next_state`, `terminal`) experienced by the agent during training. These transitions are then sampled randomly to train the agent's neural network.

## Features

- Efficient storage and retrieval of transitions
- Customizable buffer size
- Support for prioritized experience replay (PER)
- Support for both flat and image observations spcaes
- Support for both continuous and discrete action spaces
- Easy integration with deep reinforcement learning algorithms

## Installation

To use this module in your project, simply include the `ExperienceReplay.jl` file in your project directory and import it using:

```julia
include("ExperienceReplay.jl")
using .ExperienceReplay
```

or

Install with Pkg:

```julia
using Pkg
Pkg.add(url="https://github.com/p-w-rs/ExperienceReplay.jl")

using ExperienceReplay
```

## Usage

Create a new experience replay buffer with the desired state and action dimensions, along with the maximum buffer size:

```julia
buffer = Buffer(state_dim, action_dim, max_size; discrete=true)  # for discrete action space
buffer = Buffer(state_dim, action_dim, max_size; discrete=false)  # for continuous action space
```

state_dim is expected to be a tuple of dimensions for the state space, e.g. (4,) for a flat state space or (84, 84) or (84, 84, 3) whereas action_dim is expected to be a single integer for the action space. Also, if you use a 2d space such as (84, 84) the buffer will automatically convert itto a 3d space (84, 84, 1) to be compatible with general usage of neural networks.

Store a transition in the buffer:

```julia
store!(buffer, state, action, reward, next_state, terminal; p=priority)
```

Sample a batch of transitions from the buffer:

```julia
states, actions, rewards, next_states, terminals = get!(buffer, batch_size)
```

Update the priorities of the sampled transitions (for prioritized experience replay):

```julia
setp!(buffer, priorities)
```

Reset the priorities of all transitions in the buffer:

```julia
resetp!(buffer; p=default_priority)
```

## Contributing

Contributions to improve the functionality or efficiency of the replay buffer are welcome. Please submit a pull request with your proposed changes.
