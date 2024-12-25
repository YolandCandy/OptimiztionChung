import random

def Input():
    T, N, M = map(int, input().split())
    class_subjects = []
    for i in range(N):
        class_subjects.append(list(map(int, input().split()))[:-1])

    teacher_can_teach = []
    for i in range(T):
        teacher_can_teach.append(list(map(int, input().split()))[:-1])

    number_lesson = list(map(int, input().split()))
    return T, N, M, class_subjects, teacher_can_teach, number_lesson

def fitness(individual, T, N, M, teacher_can_teach, number_lesson, class_subjects):
    assigned_classes = 0
    teacher_timetable = {t: [0] * 60 for t in range(1,T + 1)}
    class_timetable = {c: [0] * 60 for c in range(1, N + 1)}
    
    for c, s, start_slot, t in individual:
        duration = number_lesson[s - 1]
        valid = True  

        for slot in range(start_slot, start_slot + duration):
            if slot >= 60 or class_timetable[c][slot] == 1:
                valid = False            
                break 

        for slot in range(start_slot, start_slot + duration):
            if slot >= 60 or teacher_timetable[t][slot] == 1:
                valid = False                     
                break 

        if valid:
            assigned_classes += 1
            for slot in range(start_slot, start_slot + duration):
                teacher_timetable[t][slot] = 1
                class_timetable[c][slot] = 1
    return assigned_classes

def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    probability = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, probability, k=2)
    return parents                       

def generate_individual(T, N, teacher_can_teach, class_subjects):
    schedule = []
    for c, subjects in enumerate(class_subjects, start = 1):
        for s in subjects:
            t = random.choice([i + 1 for i in range(T) if s in teacher_can_teach[i]])
            start_slot = random.randint(1, 60)  
            schedule.append((c, s, start_slot, t))
    return schedule
        
def initialize_population(T, N, teacher_can_teach, class_subjects):
    return [generate_individual(T, N, teacher_can_teach, class_subjects) for _ in range(POPULATION_SIZE)]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, len(individual) - 1)
        c, s, t, start_slot = individual[i]
        new_t = random.choice([i + 1 for i in range(T) if s in teacher_can_teach[i]])
        new_start_slot = random.randint(1, 59)
        individual[i] = (c, s, new_start_slot, new_t,)
    return individual

def genetic_algorithm():
    population = initialize_population(T, N, teacher_can_teach, class_subjects)
    best_individual = None
    best_fitness = 0
    
    for generation in range(GENERATIONS):
        fitnesses = [fitness(ind, T, N, M, teacher_can_teach, number_lesson, class_subjects) for ind in population]
        best_generation_fitness = max(fitnesses)
        best_generation_individual = population[fitnesses.index(best_generation_fitness)]
        
        if best_generation_fitness > best_fitness:
            best_fitness = best_generation_fitness
            best_individual = best_generation_individual
        
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))
        
        population = new_population
    
    return best_individual, best_fitness

# Main Execution
T, N, M, class_subjects, teacher_can_teach, number_lesson = Input()

POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.05  # Adjust for better diversity
best_solution, best_fitness = genetic_algorithm()

print(f"Best Fitness: {best_fitness}")
print("Schedule:")
for entry in best_solution:
    print(' '.join(map(str, entry)))