import time
from mater.network import MatcherNetwork
from mater.ranks import *
from fire import Fire

def main():
    #ranks = random_arrays
    ranks = one_collision
    network = MatcherNetwork(ranks, 100, 100, 90, 110, 0.2)
    no_update_count = 0
    while no_update_count < 10:
        print(network.neurons)

        was_updated = network.random_epoch()
        if was_updated:
            no_update_count = 0
        else:
            no_update_count += 1
        time.sleep(0.1)

    print('result:')
    print(network.neurons)
    print('ranking:')
    print(ranks)
    score = numpy.sum(ranks * network.neurons)
    print('score:', score)



if __name__ == '__main__':
    Fire(main)
