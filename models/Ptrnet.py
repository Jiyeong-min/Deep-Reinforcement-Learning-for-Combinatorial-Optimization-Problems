# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V = nn.Linear(self.hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, enc_outputs, dec_output, mask):
        w1_e = self.W1(enc_outputs)
        w2_d = self.W2(dec_output)
        tanh_output = self.tanh(w1_e + w2_d)
        v_dot_tanh = self.V(tanh_output).squeeze(2)
        # masking
        v_dot_tanh += mask
        attention_weights = F.softmax(v_dot_tanh, dim=1)
        return attention_weights

class Encoder(nn.Module):
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.cell = nn.GRU(self.input_dim, self.hidden_dim, 1, batch_first=True)

    def forward(self, input):
        enc_output, enc_hidden_state = self.cell(input)
        return enc_output, enc_hidden_state

class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.cell = nn.GRU(self.input_dim, self.hidden_dim, 1, batch_first=True)
        self.attention_layer = Attention(self.hidden_dim)

    def forward(self, input, enc_output, hidden_state, pointer, mask):
        idx = pointer.repeat(1, 2).unsqueeze(1)
        dec_output, dec_hidden = self.cell(input.gather(1, idx), hidden_state)
        attention_weights = self.attention_layer(enc_output, dec_output, mask)

        return attention_weights, dec_hidden

class PtrNet(nn.Module):
    def __init__(self, hidden_dim, input_dim=2, deterministic=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.deterministic = deterministic

        self.encoder = Encoder(self.hidden_dim)
        self.decoder = Decoder(self.hidden_dim)

    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(1)
        if input.is_cuda:
            probs = torch.zeros(batch_size, 1, device=torch.device('cuda'))
            pointers = torch.zeros(batch_size, 1, dtype=torch.long, device=torch.device('cuda'))
            mask = torch.zeros(batch_size, seq_len, dtype=torch.float, device=torch.device('cuda'))

            pointer = torch.zeros(batch_size, 1, dtype=torch.long, device=torch.device('cuda'))
        else:
            probs = torch.ones(batch_size, 1)
            pointers = torch.zeros(batch_size, 1)
            mask = torch.zeros(batch_size, seq_len)

            pointer = torch.zeros(batch_size, 1, dtype=torch.long)

        # Encoding
        enc_output, enc_hidden_state = self.encoder(input)

        mask = self.update_mask(mask, pointer)

        # Decoding
        for i in range(seq_len - 1):
            if i == 0:
                attention_weights, dec_hidden_state = self.decoder(input, enc_output, enc_hidden_state, pointer, mask)
            else:
                attention_weights, dec_hidden_state = self.decoder(input, enc_output, dec_hidden_state, pointer, mask)

            if self.deterministic:
                prob, pointer = torch.max(attention_weights, dim=1)
                mask = self.update_mask(mask, pointer)
                prob = prob.unsqueeze(1)
                pointer = pointer.unsqueeze(1)
            else:
                pointer = attention_weights.multinomial(1, replacement=True)
                prob = torch.gather(attention_weights, 1, pointer)
                mask = self.update_mask(mask, pointer)

            probs += torch.log(prob)
            pointers = torch.cat([pointers, pointer], dim=1)

        return probs, pointers

    def update_mask(self, mask, pointer):
        for batch, i in enumerate(pointer):
            mask[batch, i] = float('-inf')
        return mask

    def get_length(self, input, solution):
        # Convert solution to dtype int64
        solution = solution.long()

        current_coords = torch.gather(input, 1, solution.unsqueeze(-1).expand(-1, -1, 2))
        next_coords = torch.roll(current_coords, -1, dims=1)
        distances = torch.sqrt(torch.sum((current_coords - next_coords) ** 2, dim=-1))
        tour_length = torch.sum(distances, dim=1)

        return tour_length.unsqueeze(1)

class Critic(nn.Module):
    def __init__(self, hidden_dim, input_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.encoder = Encoder(self.hidden_dim)
        self.decoder_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_2 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, input):
        enc_output, enc_hidden_state = self.encoder(input)
        dec_hidden_state = self.decoder_1(enc_hidden_state)
        dec_hidden_state = self.relu(dec_hidden_state)
        dec_output = self.decoder_2(dec_hidden_state)
        return dec_output.squeeze(0)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define your PtrNet and Critic
# ...

# Define your Prize-Collecting Environment
class PCTSPEnvironment:
    def __init__(self, num_cities, max_reward, max_budget):
        self.num_cities = num_cities
        self.max_reward = max_reward
        self.max_budget = max_budget
        self.reset()

    def reset(self):
        self.city_coordinates = torch.rand((self.num_cities, 2))
        self.city_rewards = torch.randint(1, self.max_reward, (self.num_cities,))
        self.budget = torch.randint(1, self.max_budget, (1,)).item()
        self.current_city = 0  # Starting from city 0
        self.visited_cities = set([0])

        # Print random data
        print("Random Data for Current Episode:")
        print("City Coordinates:\n", self.city_coordinates)
        print("City Rewards:\n", self.city_rewards)
        print("Budget:\n", self.budget)

    def get_state(self):
        return self.city_coordinates

    def step(self, action):
        next_city = action.item()
        reward = 0

        # Check if the next city is within budget and not visited
        if next_city not in self.visited_cities and reward + self.city_rewards[next_city] <= self.budget:
            reward += self.city_rewards[next_city]
            self.budget -= self.city_rewards[next_city]
            self.visited_cities.add(next_city)
            self.current_city = next_city

        done = len(self.visited_cities) == self.num_cities

        return self.get_state(), reward, done

# Training Loop
num_cities = 100
max_reward = 100
max_budget = 200
hidden_dim = 256
lr = 0.0001

env = PCTSPEnvironment(num_cities, max_reward, max_budget)
dataset = TensorDataset(env.get_state().unsqueeze(0))
dataloader = DataLoader(dataset, batch_size=64)

model = PtrNet(hidden_dim=hidden_dim)
critic = Critic(hidden_dim=hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)

print("Start training...")

for epoch in range(100):
    for i, s_i in tqdm(enumerate(dataloader)):
        s_i = s_i[0]

        # Sample a tour from PtrNet
        p, pi = model(s_i)
        L = model.get_length(s_i, pi)
        b = critic(s_i)

        with torch.no_grad():
            advantage = L - b

        # Reward function: maximize collected reward while minimizing tour length
        reward = torch.mean(advantage * p)

        optimizer.zero_grad()
        loss = -reward  # We use negative reward as PtrNet is trained for minimization
        loss.backward()
        optimizer.step()

        # Update Critic
        optimizer_critic.zero_grad()
        loss_critic = F.mse_loss(L, b)
        loss_critic.backward()
        optimizer_critic.step()

        if (i + 1) % 1 == 0:
          print(f"Epoch: {epoch + 1}, Step: {i + 1}, Reward: {reward.item()}")


print("Training completed.")


# %%
#pip install tqdm

# %%
def generate_travel_path(model, env):
    model.eval()
    with torch.no_grad():
        current_state = env.get_state().unsqueeze(0)  # unsqueeze(0) 추가
        tour_prob, tour_indices = model(current_state)
        tour_length = model.get_length(current_state, tour_indices)

    return tour_indices.squeeze(0).tolist(), tour_length.item()  # squeeze(0) 추가

# ...

# After training
num_test_episodes = 5  # Adjust as needed

print("Testing...")

for episode in range(num_test_episodes):
    env.reset()
    travel_path, total_reward = generate_travel_path(model, env)


    print(f"Episode {episode + 1} - Travel Path: {travel_path}, Total Reward: {total_reward}")

print("Testing completed.")


# %%
import matplotlib.pyplot as plt

# def visualize_travel_path(city_coordinates, travel_path):
#     x = city_coordinates[:, 0]
#     y = city_coordinates[:, 1]

#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, c='red', marker='o', label='Cities')

#     # Convert travel_path to integers
#     travel_path = [int(city) for city in travel_path]

#     for i in range(len(travel_path) - 1):
#         start_city = travel_path[i]
#         end_city = travel_path[i + 1]
#         plt.plot([x[start_city], x[end_city]], [y[start_city], y[end_city]], 'b-')

#     # Connect the last and first city
#     plt.plot([x[travel_path[-1]], x[travel_path[0]]], [y[travel_path[-1]], y[travel_path[0]]], 'b-')
#     plt.title('Travel Path Visualization')
#     plt.xlabel('X-coordinate')
#     plt.ylabel('Y-coordinate')
#     plt.legend()
#     plt.show()

# ...

# After training
num_test_episodes = 1  # Adjust as needed

print("Testing...")

for episode in range(num_test_episodes):
    env.reset()
    travel_path, total_reward = generate_travel_path(model, env)

    print(f"Episode {episode + 1} - Travel Path: {travel_path}, Total Reward: {total_reward}")

    # Visualize the travel path
    # visualize_travel_path(env.city_coordinates.numpy(), travel_path)

print("Testing completed.")


# 추가
torch.save(model.state_dict(), 'tsp_100_model_weigths_v2.pth')