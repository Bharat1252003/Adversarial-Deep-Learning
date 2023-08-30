import torch
import torch.nn as nn
import torch.optim as optim

from GA import *
from train_test_split import *
from config import *
from CNN_unchanged import *

train_loader_orig, test_loader_orig = train_test_split(data_dir)

# start CNN_orig
CNN_orig = create_model()
criterion = nn.NLLLoss()
optimizer = optim.Adam(CNN_orig.parameters(), lr = learning_rate)
print("start")
CNN_orig = train_loop(train_loader_orig, test_loader_orig, model=CNN_orig, optimizer=optimizer, criterion=criterion)
print("start")
torch.save(CNN_orig.state_dict(), r'C:\models\CNN_adversarial_trained_new.pth')
# end CNN orig

maxpayoff = 0
exitloop = False

alpha_pop = [
    PopObj(0.01*np.random.random((3,100,100))) for _ in range(pop_size)
]

train_loader_new = train_loader_orig
gen = 0  #generation

while gen < max_iter and not exitloop:
    gen+=1
    print(f"Number of alphas: {len(alpha_pop)}")
    alpha_sorted = fitness(train_loader_new, alpha_pop, CNN_orig)
    alpha_best = alpha_sorted[0]
    print(f"Best fitness: {alpha_best.fitness}\n")
    save_image(alpha_best.image, f"alpha_{gen}_3")

    CNN_temp = create_model()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(CNN_temp.parameters(), lr = learning_rate)
    data_adversarial = new_data(train_loader_orig, test_loader_orig, alpha_best.image)
    train_loader_new, test_loader_new = data_adversarial[0]
    
    save_dataloader_img(train_loader_new, f"ad_{gen}_3")

    assert type(train_loader_new) == torch.utils.data.DataLoader
    assert type(test_loader_new) == torch.utils.data.DataLoader

    print("New data created")

    CNN_temp = train_loop(train_loader_new, test_loader_new, optimizer=optimizer, criterion=criterion, model=CNN_temp)
    
    print("New CNN trained")

    if abs(maxpayoff-alpha_best.fitness) < 1:
        curr_payoff = alpha_best.fitness
        print(f'Curr_payoff: {curr_payoff}, max_payoff: {maxpayoff}, difference: {curr_payoff-maxpayoff}')
        maxpayoff = alpha_best.fitness

        parents, offsprings = selection(alpha_pop)

        for child1, child2 in zip(offsprings[:len(offsprings)//2], offsprings[len(offsprings)//2:]): # randomize more, numpy select randomly/acc to a dist
            child1, child2 = crossover_img(child1, child2)

        for mutant in offsprings:
            mutant = mutation(mutant)
        
        alpha_pop = np.append(parents, offsprings)

    torch.save(CNN_orig.state_dict(), r'C:\models\CNN_adversarial_trained_new.pth')