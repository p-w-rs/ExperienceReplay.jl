# ExperienceReplay.jl

This module provides an implementation of a replay buffer, a fundamental component in off-policy reinforcement learning algorithms. The replay buffer stores transitions (state, action, reward, next_state, terminal) and supports prioritized experience replay.

## Features

- Efficient storage and retrieval of transitions
- Support for prioritized experience replay (PER)
- Handles both flat and image observation spaces
- Supports both continuous and discrete action spaces
- Customizable buffer size

## Installation

Include the `ExperienceReplay.jl` file in your project directory and import it:

```julia
include("ExperienceReplay.jl")
using .ExperienceReplay
```

Or install with Pkg:

```julia
using Pkg
Pkg.add("ExperienceReplay")
```

## Usage

### Creating a Buffer

Create a new experience replay buffer with the desired state and action dimensions, along with the maximum buffer size:

```julia
# For discrete action space
buffer = Buffer(state_dim, action_dim, max_size; discrete=true)

# For continuous action space
buffer = Buffer(state_dim, action_dim, max_size; discrete=false)
```

- `state_dim`: Tuple of dimensions for the state space
  - For flat state space: e.g., `(784,)`
  - For image state space: e.g., `(28, 28)` or `(28, 28, 3)`
- `action_dim`: Integer representing the action space dimension
- `max_size`: Maximum number of transitions to store in the buffer

Note: If you use a 2D state space like `(28, 28)`, the buffer will automatically convert it to a 3D space `(28, 28, 1)`.

Note: This is a ciruclar buffer so if the `max_size` is reached the buffer will start to overwirte entries by looping back to the start

### Storing Transitions

Store a transition in ` the buffer:

```julia
store!(buffer, state, action, reward, next_state, terminal; p=priority)
```

- `p` (optional): Priority of the transition (default is 1.0f0)
  - Higher priority values make the sample more likely to be selected

### Sampling Transitions

Sample a batch of transitions from the buffer:

```julia
states, actions, rewards, next_states, terminals = get_batch!(buffer, batch_size)
```

### Updating Priorities

Update the priorities of the last sampled batch:

```julia
setp!(buffer, new_priorities)
```

- `new_priorities`: Vector of new priority values for the last sampled batch
- This function assumes the priorities correspond to the last batch pulled from the replay

### Resetting Priorities

Reset the priorities of all transitions in the buffer:

```julia
resetp!(buffer; p=default_priority)
```

- `p` (optional): New default priority value (default is 1.0f0)

## Important Notes on Priorities

- Higher priority values make samples more likely to be selected during `get_batch!`
- When updating priorities with `setp!`, it assumes the new priorities correspond to the last batch sampled from the buffer
- Use `resetp!` to set all priorities to a specific value, useful for resetting the buffer's prioritization

## Contributing

Contributions to improve the functionality or efficiency of the replay buffer are welcome. Please submit a pull request with your proposed changes.
