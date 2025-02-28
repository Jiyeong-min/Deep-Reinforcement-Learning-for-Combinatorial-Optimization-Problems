import math
import numpy as np
import random
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch



class Tsp:
    
    def __init__(self, N):
        self.N = N
    
    def next_batch(self, batch_size=1, solve=False):
        X, Y = [], []
        for b in range(batch_size):
            data = self.generate_data(solve=solve)
            points = data['points']
            if solve:
                solved = self.solve_tsp(data)
                Y.append(solved)
            X.append(points)
        
        return torch.tensor(np.asarray(X),dtype=torch.float), torch.tensor(np.asarray(Y),dtype=torch.float)
    
    def length(self, x, y):
        return (math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))
    
    
    def generate_data(self, solve=False):
        data = {}
        radius = 1
        rangeX = (0, 10)
        rangeY = (0, 10)
        qty = self.N
        
        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius:
                    deltas.add((x,y))
                    
        randPoints = []
        excluded = set()
        i = 0
        while i < qty:
            x = random.uniform(rangeX[0], rangeX[1])
            y = random.uniform(rangeY[0], rangeY[1])
            if (x,y) in excluded:
                continue
            randPoints.append((x,y))
            i += 1
            excluded.update((x+dx, y+dy) for (dx, dy) in deltas)
        
        data['points'] = randPoints
        
        if not solve:
            return data
        else:
            distance = np.zeros((self.N,self.N))
            for i in range(len(randPoints)):
                for j in range(len(randPoints)):
                    distance[i][j] = self.length(randPoints[i], randPoints[j])

            data['distance_matrix'] = distance
            data['num_vehicle'] = 1 # num_vehicle 
            data['depot'] = 0

            return data
    
    
    def solve_tsp(self, data):
        
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicle'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        solution = routing.SolveWithParameters(search_parameters)
        
        sol_list = []
        index = routing.Start(0)
        sol_list.append(index)
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            sol_list.append(index)

        return sol_list[:-1]
    
    
if __name__ == "__main__":
    p = Tsp()
    X, Y = p.next_batch(1)
    print(X)
    print(Y)


