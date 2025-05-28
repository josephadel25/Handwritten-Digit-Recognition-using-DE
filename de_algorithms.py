from tensorflow.keras import mixed_precision
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
mixed_precision.set_global_policy('float32')
import tensorflow as tf
import numpy as np
from model_utils import build_model, flatten_weights, set_model_weights
import random
with tf.device('/GPU:0'):

    # Initialization
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    def gen_individual_random(seed):
        model = build_model(seed=seed)
        return flatten_weights(model)

    def gen_individual_gaussian_noise(seed , base=None):
        if base is None:
            base = gen_individual_random(seed)
        noise = np.random.normal(0, 0.05, size=base.shape)
        return base + noise

    def gen_population(pop_size, global_model, x_val, y_val, strategy="random"):
        population = []
        fitnesses = []
        if strategy == "random":
            for _ in range(pop_size):
                ind = gen_individual_random(np.random.randint(0, 1e6))
                population.append(ind)
                fitnesses.append(evaluate_fitness(ind,global_model, x_val, y_val))

        elif strategy == "gaussian":
            base = gen_individual_random(np.random.randint(0, 1e6) )
            for _ in range(pop_size):
                ind = gen_individual_gaussian_noise(np.random.randint(0, 1e6) , base)
                population.append(ind)
                fitnesses.append(evaluate_fitness(ind,global_model, x_val, y_val))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return population, fitnesses

    # Evaluation
    @tf.function
    def fast_eval(model, x, y):
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = tf.argmax(y, axis=1) 
        logits = model(x, training=False)
        acc = tf.keras.metrics.sparse_categorical_accuracy(y, logits)
        return tf.reduce_mean(acc)

    def fast_evaluate(model, individual, x, y):
        set_model_weights(model, individual)
        return fast_eval(model, x, y)
    
    def evaluate_fitness(individual, model, x_val, y_val):
        try:
            accuracy = fast_evaluate(model, individual, x_val, y_val).numpy()
            return float(accuracy)
        except Exception as e:
            print("Evaluation error:", str(e))
            return 0.0

    # Operators
    def mutation_rand_1(population, current_idx, F):
        others = list(range(len(population)))
        others.remove(current_idx)
        a, b, c = np.random.choice(others, 3, replace=False)
        mutant = population[a] + F * (population[b] - population[c])
        return np.clip(mutant, -1, 1)

    def mutation_best_1(population, fitnesses, F):
        best = population[np.argmax(fitnesses)]
        others = list(range(len(population)))
        others.remove(np.argmax(fitnesses))
        a, b = np.random.choice(others, 2, replace=False)
        mutant = best + F * (population[a] - population[b])
        return np.clip(mutant, -1, 1)

    def crossover_binomial(target, mutant, CR):
        trial = []
        for i in range(len(target)):
            if np.random.rand()  < CR:
                trial.append(mutant[i])        
            else:                               
                trial.append(target[i])         
        return np.array(trial)

    def crossover_exponential(target, mutant, CR):
        size = len(target)
        start = np.random.randint(size)
        length = 0
        while np.random.rand() < CR and length < size:
            length += 1
        indices = [(start + i) % size for i in range(length)]
        trial = np.copy(target)
        trial[indices] = mutant[indices]
        return trial

    # Selection
    def select_better(target, trial, fitness_target, fitness_function, x_val, y_val,global_model):
        fitness_trial = fitness_function(trial,global_model, x_val, y_val)
        if fitness_trial > fitness_target:
            return trial, fitness_trial
        else:
            return target, fitness_target

    def select_tournament(target, trial, fitness_target, fitness_function, x_val, y_val,global_model):
        fitness_trial = fitness_function(trial,global_model, x_val, y_val)
        total = fitness_trial + fitness_target

        if total == 0:
            prob_trial = 0.5
        else:
            prob_trial = fitness_trial / total
        if np.random.rand() < prob_trial:
            return trial, fitness_trial
        else:
            return target, fitness_target

    def selection_crowding(population, fitnesses, trial, trial_fitness):
        distances = np.linalg.norm(population - trial, axis=1)
        most_similar_idx = np.argmin(distances)
        if trial_fitness > fitnesses[most_similar_idx]:
            return trial, trial_fitness, most_similar_idx
        return population[most_similar_idx], fitnesses[most_similar_idx], None

    def differential_evolution(global_model, population, fitnesses, generations, mutation_func, crossover_func,
                            F, CR, x_val, y_val, selection_func, progress_callback=None):
        best_fitness_progress = []
        best_so_far = max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]

        no_improvement_generations = 0
        patience = 100

        for gen in range(generations):
            for i in range(len(population)):
                # Mutation
                mutant = mutation_func(population, i, F) if mutation_func.__name__ == "mutation_rand_1" \
                    else mutation_func(population, fitnesses, F)

                # Crossover
                trial = crossover_func(population[i], mutant, CR)
                trial_fitness = evaluate_fitness(trial, global_model, x_val, y_val)

                # Selection
                if selection_func.__name__ == "selection_crowding":
                    selected, selected_fitness, replaced_idx = selection_func(population, fitnesses, trial, trial_fitness)
                    if replaced_idx is not None:
                        population[replaced_idx] = selected
                        fitnesses[replaced_idx] = selected_fitness
                else:
                    selected, selected_fitness = selection_func(
                        population[i], trial, fitnesses[i], evaluate_fitness, x_val, y_val, global_model)
                    population[i] = selected
                    fitnesses[i] = selected_fitness

            gen_best_index = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_index]
            gen_best_individual = population[gen_best_index]
            best_fitness_progress.append(gen_best_fitness)

            # Track best-so-far and improvement
            if gen_best_fitness > best_so_far:
                best_so_far = gen_best_fitness
                best_individual = gen_best_individual
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            if progress_callback:
                progress_callback(gen + 1, gen_best_fitness)

            print(f"Generation of DE {gen+1} - Best Accuracy: {gen_best_fitness:.4f}")

            if gen > 500 and (np.std(fitnesses) < 0.005 or no_improvement_generations >= patience):
                print(f"Stopped early at generation {gen+1} due to stagnation.")
                break

        return best_individual, best_fitness_progress
