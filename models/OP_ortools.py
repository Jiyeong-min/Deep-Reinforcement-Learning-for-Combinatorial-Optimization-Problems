import time
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import pywrapcp
from itertools import combinations

# 주어진 데이터
custom_city_coordinates = [[4.1113e-02, 2.7530e-01],
        [3.3297e-01, 9.0903e-01],
        [1.3806e-01, 8.5710e-01],
        [6.0656e-01, 3.1280e-01],
        [1.2300e-01, 8.9359e-01],
        [9.7994e-01, 3.3151e-01],
        [6.0642e-01, 3.6744e-02],
        [6.8014e-01, 6.8255e-01],
        [3.1067e-01, 7.9592e-01],
        [1.7952e-01, 4.6431e-01],
        [3.5741e-01, 8.0906e-01],
        [9.1592e-01, 2.8179e-01],
        [1.3437e-02, 7.5215e-01],
        [4.4005e-01, 2.4348e-01],
        [2.2654e-02, 4.2046e-02],
        [9.5371e-01, 6.5710e-01],
        [2.8128e-01, 3.2654e-01],
        [4.5201e-01, 9.5240e-01],
        [9.2409e-01, 5.5540e-01],
        [6.6416e-01, 6.7016e-01],
        [9.0091e-01, 8.6161e-01],
        [5.2090e-01, 2.2549e-01],
        [6.2680e-01, 5.2165e-01],
        [5.1865e-01, 3.6116e-01],
        [3.1807e-01, 8.7791e-01],
        [5.4913e-01, 1.6708e-02],
        [9.4732e-01, 6.6145e-01],
        [6.2318e-01, 8.5188e-01],
        [4.5631e-01, 8.0021e-01],
        [8.7411e-01, 7.9148e-01],
        [5.9205e-01, 9.2241e-02],
        [5.6950e-01, 2.2686e-01],
        [4.1965e-01, 2.7034e-01],
        [6.0782e-01, 8.5491e-01],
        [8.1893e-01, 5.6939e-01],
        [8.5509e-01, 1.6715e-01],
        [2.1408e-01, 3.8682e-01],
        [5.2779e-01, 9.7305e-02],
        [8.9995e-01, 8.8410e-01],
        [5.2896e-01, 3.7676e-01],
        [7.5562e-01, 8.7351e-01],
        [8.0516e-01, 3.9159e-01],
        [3.9559e-01, 5.5162e-01],
        [9.9175e-02, 6.7559e-02],
        [9.5217e-01, 3.1480e-01],
        [7.6946e-01, 4.1602e-01],
        [2.4200e-01, 6.2740e-01],
        [2.0175e-01, 2.5439e-01],
        [9.6672e-01, 7.0831e-01],
        [7.2874e-01, 7.4854e-03],
        [1.6811e-01, 6.9209e-01],
        [8.7201e-01, 4.2800e-01],
        [6.4685e-01, 9.0233e-01],
        [6.0125e-01, 2.9764e-01],
        [2.9087e-05, 5.6494e-01],
        [8.2040e-01, 2.4262e-01],
        [2.0714e-01, 3.9125e-01],
        [2.3844e-01, 5.1435e-01],
        [8.1320e-02, 5.5446e-01],
        [7.9390e-02, 5.8096e-01],
        [4.5412e-01, 9.5256e-01],
        [3.6516e-01, 6.1938e-01],
        [5.8235e-01, 6.0247e-01],
        [7.8591e-01, 3.1591e-01],
        [1.1351e-01, 1.2458e-01],
        [4.1672e-01, 2.4686e-01],
        [8.4142e-01, 5.7256e-01],
        [1.7300e-01, 9.5264e-01],
        [2.9269e-01, 6.6284e-01],
        [9.2616e-01, 7.8410e-01],
        [1.8073e-02, 8.2971e-01],
        [7.4459e-01, 4.2912e-01],
        [2.7218e-01, 7.4443e-01],
        [5.5749e-01, 2.8659e-01],
        [5.3267e-01, 8.1997e-02],
        [2.6358e-02, 6.7174e-01],
        [6.4116e-02, 7.6069e-01],
        [9.5384e-01, 6.1731e-01],
        [7.7848e-01, 7.8237e-02],
        [2.8993e-01, 3.4746e-01],
        [6.6936e-01, 4.8335e-01],
        [5.1635e-01, 1.5132e-01],
        [7.9474e-01, 5.6714e-01],
        [7.5092e-01, 3.6046e-01],
        [5.6074e-01, 6.0465e-01],
        [3.4087e-01, 4.4652e-01],
        [8.6067e-01, 1.0668e-01],
        [8.8831e-01, 1.0026e-01],
        [2.5171e-01, 1.8505e-01],
        [1.8670e-01, 9.9595e-01],
        [4.0885e-01, 9.5423e-01],
        [1.2225e-01, 8.7368e-01],
        [6.5858e-01, 5.1979e-01],
        [5.5320e-01, 8.0656e-01],
        [4.9888e-01, 8.5352e-01],
        [4.9201e-01, 3.3716e-01],
        [2.0759e-01, 8.5694e-01],
        [2.3280e-02, 9.4431e-01],
        [9.9073e-01, 9.2489e-01],
        [4.8221e-01, 3.2881e-01]]

custom_city_rewards = [54, 71, 18, 67, 77, 52, 39, 47, 46, 66, 81, 22, 41, 23, 81, 14, 16, 25,
        79, 11, 39, 98, 33, 22, 80,  6, 15, 79, 46, 61, 43, 41,  3, 25, 86, 62,
        28, 95, 41, 84, 43, 59, 12, 11, 79, 20, 61, 60, 29, 63, 22, 61,  6, 98,
        11, 37, 33, 15, 35,  2, 78, 70, 33, 17, 71, 68, 64, 34, 24, 49, 36, 62,
        32, 21, 96, 15, 42, 79, 19, 75, 88, 53, 56, 23, 19, 75, 16, 99, 85, 42,
        40, 42, 96, 90, 36, 11, 73, 94, 26, 24]

# Maximum travel distance
max_travel_distance = 50

def create_data_model():
    data = {}
    data['locations'] = custom_city_coordinates
    data['num_vehicles'] = 1
    data['depot'] = 0  # Assuming the starting point is the first city

    return data

def calculate_distance(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

def create_distance_callback(data):
    def distance_callback(from_node, to_node):
        return calculate_distance(data['locations'][from_node], data['locations'][to_node])

    return distance_callback

import time
from ortools.constraint_solver import pywrapcp

# 이전 함수들은 그대로 유지합니다.

def solve_orienteering(custom_city_coordinates, custom_city_rewards, max_travel_distance):
    start_time = time.time()  # 시작 시간 기록

    data = create_data_model()

    # Greedy selection of nodes based on rewards and distance
    selected_nodes = []
    current_node = data['depot']
    remaining_distance = max_travel_distance

    while True:
        max_reward = -1
        next_node = None

        for node in range(len(custom_city_coordinates)):
            if node not in selected_nodes and remaining_distance >= calculate_distance(data['locations'][current_node], data['locations'][node]):
                if custom_city_rewards[node] > max_reward:
                    max_reward = custom_city_rewards[node]
                    next_node = node

        if next_node is not None:
            selected_nodes.append(next_node)
            remaining_distance -= calculate_distance(data['locations'][current_node], data['locations'][next_node])
            current_node = next_node
        else:
            break

    # Calculate total reward and total distance for the selected nodes
    total_reward = sum(custom_city_rewards[node] for node in selected_nodes)
    total_distance = sum(calculate_distance(data['locations'][selected_nodes[i]], data['locations'][selected_nodes[i+1]]) for i in range(len(selected_nodes) - 1))

    end_time = time.time()  # 종료 시간 기록
    calculation_time = end_time - start_time  # 계산에 걸린 시간 계산

    return total_reward, total_distance, calculation_time

# Code execution
total_reward, total_distance, calculation_time = solve_orienteering(custom_city_coordinates, custom_city_rewards, 10)

if total_reward is not None:
    print(f"Total Reward: {total_reward}")
    print(f"Total Travel Distance: {total_distance}")
    print(f"Calculation Time: {calculation_time} seconds")
else:
    print("Could not find a valid route within the distance constraint.")

