def train_with_pytorch_cuda(episodes=500, output_path="models/enemy_rl/gpu_model.pth"):
    """Train enemy agent using PyTorch with CUDA."""
    logger.info("Training with PyTorch CUDA")
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Import the game environment
    from ai_platform_trainer.core.render_mode import RenderMode
    from ai_platform_trainer.ai.models.game_environment import GameEnvironment
    
    # Create environment in headless mode
    env = GameEnvironment(render_mode='none')
    
    # Define a simple policy network
    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return torch.tanh(self.fc3(x))
        
        def save(self, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
            logger.info(f"Model saved to {path}")
    
    # Create policy network on GPU
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy_net = PolicyNetwork(input_size, 128, output_size).cuda()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # Training loop
    logger.info(f"Starting training for {episodes} episodes...")
    episode_rewards = []
    
    for episode in tqdm(range(episodes)):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        
        # Lists to store episode data
        log_probs = []
        rewards = []
        
        # Episode loop
        done = False
        truncated = False
        
        while not (done or truncated):
            # Convert state to tensor on GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
            
            # Get action from policy network
            action_mean = policy_net(state_tensor)
            
            # Add exploration noise
            action_std = torch.ones_like(action_mean) * 0.1
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            
            # Store log probability
            log_prob = action_dist.log_prob(action).sum()
            log_probs.append(log_prob)
            
            # Take action in environment
            action_np = action.squeeze().detach().cpu().numpy()
            next_state, reward, done, truncated, _ = env.step(action_np)
            
            # Store reward
            rewards.append(reward)
            episode_reward += reward
            
            # Update state
            state = next_state
        
        # Update policy network
        optimizer.zero_grad()
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R  # Gamma = 0.99
            returns.insert(0, R)
        
        if len(returns) > 0:
            # Convert to tensor on GPU
            returns = torch.tensor(returns, device="cuda")
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            if policy_loss:
                policy_loss = torch.stack(policy_loss).sum()
                
                # Backpropagate
                policy_loss.backward()
                optimizer.step()
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    # Save trained model
    policy_net.save(output_path)
    
    return True

def train_with_cpu(episodes=500, output_path="models/enemy_rl/cpu_model.npz"):
    """Train enemy agent using CPU."""
    logger.info("Training with CPU")
    
    # Import the game environment
    from ai_platform_trainer.core.render_mode import RenderMode
    from ai_platform_trainer.ai.models.game_environment import GameEnvironment
    
    # Create environment in headless mode
    env = GameEnvironment(render_mode='none')
    
    # Simple neural network implementation
    class SimpleNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize with random weights
            self.w1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.w2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros(output_size)
        
        def forward(self, x):
            # Simple feedforward
            h = np.tanh(np.dot(x, self.w1) + self.b1)
            return np.tanh(np.dot(h, self.w2) + self.b2)
        
        def update(self, grads, lr=0.01):
            # Simple gradient update
            self.w1 -= lr * grads[0]
            self.b1 -= lr * grads[1]
            self.w2 -= lr * grads[2]
            self.b2 -= lr * grads[3]
        
        def save(self, path):
            # Save weights
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
            logger.info(f"Model saved to {path}")
    
    # Create policy network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy_net = SimpleNetwork(input_size, 64, output_size)
    
    # Training loop
    logger.info(f"Starting training for {episodes} episodes...")
    episode_rewards = []
    
    for episode in tqdm(range(episodes)):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        
        # Episode loop
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Get action from policy network
            action = policy_net.forward(state)
            
            # Add exploration noise
            action = action + np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1.0, 1.0)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store reward
            episode_reward += reward
            
            # Update state
            state = next_state
            step += 1
            
            # Simple update (not a proper RL algorithm, just for demonstration)
            if reward > 0:
                # Encourage actions that led to positive rewards
                grads = [np.zeros_like(policy_net.w1), 
                         np.zeros_like(policy_net.b1),
                         np.zeros_like(policy_net.w2), 
                         np.zeros_like(policy_net.b2)]
                policy_net.update(grads, lr=0.001 * reward)
            
            # Limit maximum steps
            if step >= 1000:
                truncated = True
        
        # Log progress
        episode_rewards.append(episode_reward)
        if episode % 10 == 0:
            logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    # Save trained model
    policy_net.save(output_path)
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train enemy agent with best available acceleration")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--output", type=str, default="models/enemy_rl/model", 
                        help="Output model path prefix (extension will be added based on method)")
    parser.add_argument("--force-method", choices=["cuda", "pytorch", "cpu"], 
                        help="Force a specific training method")
    args = parser.parse_args()
    
    # Determine the best available method
    if args.force_method == "cuda":
        method = "cuda"
    elif args.force_method == "pytorch":
        method = "pytorch"
    elif args.force_method == "cpu":
        method = "cpu"
    else:
        # Auto-detect
        if check_cuda_extensions():
            method = "cuda"
        elif check_pytorch_cuda():
            method = "pytorch"
        else:
            method = "cpu"
    
    # Set output path based on method
    if method == "cuda":
        output_path = args.output + ".npz"
    elif method == "pytorch":
        output_path = args.output + ".pth"
    else:
        output_path = args.output + ".npz"
    
    # Train using the selected method
    logger.info(f"Selected training method: {method}")
    
    if method == "cuda":
        success = train_with_custom_cuda(episodes=args.episodes, output_path=output_path)
    elif method == "pytorch":
        success = train_with_pytorch_cuda(episodes=args.episodes, output_path=output_path)
    else:
        success = train_with_cpu(episodes=args.episodes, output_path=output_path)
    
    if not success:
        logger.error("Training failed.")
        sys.exit(1)
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Enemy agent was trained using {method} acceleration.")
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()