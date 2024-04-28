import random
import heapq
from .cluster import cluster_data

def evolutionary_algorithm(evo_instance, pop_size, num_generations, mutation_rate):
    print("Initializing population")
    population = initialize_population(pop_size, len(evo_instance.columns), (evo_instance.min_clusters, evo_instance.max_clusters), evo_instance.special_columns_indices)
    print("Population created")
    best_fitness = float('-inf')
    generations_without_improvement = 0
    top_individuals = []

    for generation in range(num_generations):
        fitnesses = [fitness(evo_instance.df, 
            individual, evo_instance.columns, evo_instance.wcss_min_max, 
            evo_instance.d_min_max) 
            for individual in population]
        
        for ind, fit in zip(population, fitnesses):
            heapq.heappush(top_individuals, (-fit, ind))
            if len(top_individuals) > 5:
                heapq.heappop(top_individuals)
        
        current_best = max(fitnesses)

        if current_best > best_fitness:
            best_fitness = current_best
            generations_without_improvement = 0
            print(f"Generation {generation}: Best Fitness Improved = {best_fitness}")
        else:
            generations_without_improvement += 1
            print(f"Generation {generation}: Best Fitness = {current_best} (No Improvement)")

        if generations_without_improvement > 5:
            print("Terminating early due to no improvement.")
            break
        
        parents = select(population, fitnesses)
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                offspring.extend(crossover(parents[i], parents[i + 1]))
        population = [mutate(individual, mutation_rate, evo_instance.min_features, evo_instance.max_features, evo_instance.special_columns_indices) for individual in (parents + offspring)]

    # After all generations, print the top 5 results
    print("\nTop 5 Unique Individuals Across All Generations:")
    top_individuals.sort(reverse=True)
    for fit, individual in top_individuals:
        selected_columns = [evo_instance.df.columns[i] for i, selected in enumerate(individual[0]) if selected]
        print(f"Fitness: {-fit}, Columns: {selected_columns}, Clusters: {individual[1]}")

    return top_individuals


def initialize_population(pop_size, num_columns, cluster_range, special_columns_indices=[]):
    population = []
    for _ in range(pop_size):
        valid = False
        while not valid:
            columns = [0] * num_columns  # Initialize all columns as not selected
            num_selected_columns = random.randint(*cluster_range)  # Randomly select the number of columns to activate
            selected_indices = random.sample(range(num_columns), num_selected_columns)
            for index in selected_indices:
                columns[index] = 1

            if special_columns_indices:
                special_columns_selected = sum(columns[index] for index in special_columns_indices)
                if special_columns_selected <= 1:  # Only one of the special columns can be selected
                    valid = True
            else:
                valid = True

        clusters = random.randint(*cluster_range)   
        population.append((columns, clusters))
    return population


def fitness(df, individual, columns, wcss_min_max, d_min_max):
    selected_indices = [i for i, isSelected in enumerate(individual[0]) if isSelected]
    selected_columns = [columns[i] for i in selected_indices]
    clusters = individual[1]
    _, score = cluster_data(df, selected_columns, clusters, wcss_min_max, d_min_max)
    return score

def select(population, fitnesses):
    selected_parents = []
    tournament_size = 15
    num_parents = len(population) // 2
    
    for _ in range(num_parents):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        selected_parents.append(population[winner_index])
    return selected_parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1[0]) - 2)
    offspring1_columns = parent1[0][:crossover_point] + parent2[0][crossover_point:]
    offspring2_columns = parent2[0][:crossover_point] + parent1[0][crossover_point:]
    offspring1_clusters = parent1[1] if random.random() < 0.5 else parent2[1]
    offspring2_clusters = parent2[1] if random.random() < 0.5 else parent1[1]
    
    return [(offspring1_columns, offspring1_clusters), (offspring2_columns, offspring2_clusters)]

def mutate(individual, mutation_rate, min_features, max_features, special_columns_indices):
    num_columns = len(individual[0])
    mutated_features = individual[0].copy()

    for i in range(num_columns):
        if random.random() < mutation_rate:
            mutated_features[i] = 1 - mutated_features[i]

    num_selected = sum(mutated_features)
    if num_selected < min_features:
        needed = min_features - num_selected
        indices_to_activate = [i for i, isSelected in enumerate(mutated_features) if not isSelected]
        for index in random.sample(indices_to_activate, needed):
            mutated_features[index] = 1
    elif num_selected > max_features:
        excess = num_selected - max_features
        indices_to_deactivate = [i for i, isSelected in enumerate(mutated_features) if isSelected]
        for index in random.sample(indices_to_deactivate, excess):
            mutated_features[index] = 0

    if special_columns_indices:
        special_columns_selected = [index for index in special_columns_indices if mutated_features[index] == 1]
        while len(special_columns_selected) > 1:
            index_to_deactivate = random.choice(special_columns_selected)
            mutated_features[index_to_deactivate] = 0
            special_columns_selected.remove(index_to_deactivate)

    if random.random() < mutation_rate:
        direction = random.choice([-1, 1])
        new_cluster_count = individual[1] + direction
        new_cluster_count = max(min_features, min(new_cluster_count, max_features))
        individual = (mutated_features, new_cluster_count)
    else:
        individual = (mutated_features, individual[1])

    return individual