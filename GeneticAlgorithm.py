import random
import operator

from time import sleep
import copy

ALLELE = 27
LENGTH = 17
N = 1000
P = 0.03
BEST_N = 30
GEN = 1000

class Indivisual():
    def __init__(self):
        self.gene = []
        for i in range(LENGTH):
            self.gene.append(int(random.random() * ALLELE))
        self.fitness = 0

    def show(self):
        out_str = ''
        for str_val in self.gene:
            if str_val == 0:
                out_str += ' '
            else:
                out_str += chr(str_val + 96)
        print(out_str)

class GeneticAlgorithm():
    def __init__(self):
        self.indiv = []
        for i in range(N):
            self.indiv.append(Indivisual())
        self.ans = 'generic algorithm'

    def _compute_fitness(self, indiv):
        fit = 0
        for i, v in enumerate(self.ans):
            if v == ' ':
                fit += indiv.gene[i]
            else:
                fit += abs((indiv.gene[i] + 96) - ord(v))

        indiv.fitness = -fit + ALLELE * LENGTH

    def compute_fitness(self):
        for indiv in self.indiv:
            self._compute_fitness(indiv)

    def uniform_crossover(self, parent1, parent2):
        child = Indivisual()
        for i in range(LENGTH):
            if random.random() < 0.5:
                child.gene[i] = self.indiv[parent2].gene[i]
            else:
                child.gene[i] = self.indiv[parent1].gene[i]

        return child, None

    def mutation(self, indiv):
        if random.random() < P:
            index = int(random.random() * LENGTH)

            while True:
                g = int(random.random() * ALLELE)

                if indiv.gene[index] != g:
                    indiv.gene[index] = g
                    return

    def roulette_selection(self):
        if self.fitnesses_list is None:
            self.fitnesses_list = list(map(lambda x: x.fitness, self.indiv))
            self.total = sum(self.fitnesses_list)
        r = random.random() * self.total

        tf = 0

        for i in range(N):
            tf += self.fitnesses_list[i]
            if r < tf:
                return i

    def genChange(self):
        for i in range(GEN):
            self.compute_fitness()
            self.indiv.sort(key=operator.attrgetter('fitness'))
            self.indiv.reverse()
            print(str(i) + '世代')
            self.indiv[0].show()

            print('error = ' + str(ALLELE * LENGTH - self.indiv[0].fitness))
            if self.indiv[0].fitness == ALLELE * LENGTH:
                return 0

            # エリート
            next_indiv = copy.deepcopy(self.indiv[0:BEST_N])
            self.fitnesses_list = None
            # ルーレット・交叉
            for i in range(BEST_N, N):
                i1 = self.roulette_selection()
                i2 = i1
                while i1 == i2:
                    i2 = self.roulette_selection()

                child, _ = self.uniform_crossover(i1, i2)
                next_indiv.append(copy.deepcopy(child))

            # 突然変異
            for i in range(1, N):
                self.mutation(next_indiv[i])

            self.indiv = next_indiv


ga = GeneticAlgorithm()
ga.genChange()
