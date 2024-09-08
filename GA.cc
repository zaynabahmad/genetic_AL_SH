#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std

class TSPGeneticAlgorithm {
 public:
  TSPGeneticAlgorithm(int numPoints, int populationSize, int numGenerations,
                      double crossoverRate, double mutationRate)
      : numPoints(numPoints),
        populationSize(populationSize),
        numGenerations(numGenerations),
        crossoverRate(crossoverRate),
        mutationRate(mutationRate) {
    srand(time(0));
    initializePopulation();
  }

  void runAlgorithm() {
    for (int generation = 0; generation < numGenerations; ++generation) {
      calculateFitness();
      selectParents();
      for (int i = 0; i < populationSize; i += 2) {
        int parent1Index = i;
        int parent2Index = i + 1;
        crossover(population[parent1Index], population[parent2Index],
                  population[i], population[i + 1]);
      }
      mutate();
    }
  }

  void printBestRoute() {
    int bestIndex =
        distance(fitness.begin(), max_element(fitness.begin(), fitness.end()));
    cout << "Best Route: ";
    for (int i : population[bestIndex]) {
      cout << i << " ";
    }
    cout << endl;

    cout << "Total Distance: " << calculateDistance(population[bestIndex])
         << endl;
  }

  /**
   * will be moved again in teh private
   */

  double calculateDistance(const vector<int>& route) {  // ecluidin
    double totalDistance = 0.0;
    for (int i = 0; i < numPoints - 1; ++i) {
      int x1 = points[route[i]].first;
      int y1 = points[route[i]].second;
      int x2 = points[route[i + 1]].first;
      int y2 = points[route[i + 1]].second;
      totalDistance += sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    }
    totalDistance +=
        sqrt(pow(points[route.back()].first - points[route[0]].first, 2) +
             pow(points[route.back()].second - points[route[0]].second,
                 2));  // if i will return to the first element
    return totalDistance;
  }

  void calculateFitness() {
    fitness.resize(populationSize);
    for (int i = 0; i < populationSize; ++i) {
      fitness[i] = 1 / calculateDistance(population[i]);
    }
  }
  void selectParents() {
    vector<vector<int>> newPopulation(populationSize, vector<int>(numPoints));
    double totalFitness = accumulate(fitness.begin(), fitness.end(), 0.0);

    for (int i = 0; i + 1 < populationSize; i += 2) {
      int parent1Index = selectParentIndex(totalFitness);
      int parent2Index = selectParentIndex(totalFitness);

      crossover(population[parent1Index], population[parent2Index],
                newPopulation[i], newPopulation[i + 1]);
    }

    // Copy the new population to the original population vector
    population = newPopulation;
  }

  int selectParentIndex(double totalFitness) {
    double randomValue = rand() / (double)RAND_MAX * totalFitness;
    for (int i = 0; i < populationSize; ++i) {
      randomValue -= fitness[i];
      if (randomValue <= 0) {
        return i;
      }
    }
    std::cerr << "Error: No valid parent index found." << std::endl;
    exit(EXIT_FAILURE);
  }

  void crossover(const vector<int>& parent1, const vector<int>& parent2,
                 vector<int>& child1, vector<int>& child2) {
    int crossoverPoint1 = rand() % (numPoints - 1) + 1;
    int crossoverPoint2 =
        rand() % (numPoints - crossoverPoint1) + crossoverPoint1;

    copy(parent1.begin() + crossoverPoint1, parent1.begin() + crossoverPoint2,
         child1.begin() + crossoverPoint1);
    copy(parent2.begin() + crossoverPoint2, parent2.end(),
         child1.begin() + crossoverPoint2);

    copy(parent2.begin() + crossoverPoint1, parent2.begin() + crossoverPoint2,
         child2.begin() + crossoverPoint1);
    copy(parent1.begin() + crossoverPoint2, parent1.end(),
         child2.begin() + crossoverPoint2);
  }

  void initializePopulation() {
    population.resize(populationSize, vector<int>(numPoints));
    for (int i = 0; i < populationSize; ++i) {
      iota(population[i].begin(), population[i].end(), 0);
      random_shuffle(population[i].begin() + 1, population[i].end());
    }
  }

  void mutate() {
    for (vector<int>& route : population) {
      if (rand() / (double)RAND_MAX < mutationRate) {
        int mutationPoint1 = rand() % numPoints;
        int mutationPoint2 = rand() % numPoints;
        swap(route[mutationPoint1], route[mutationPoint2]);
      }
    }
  }

 private:
  int numPoints;
  int populationSize;
  int numGenerations;
  double crossoverRate;
  double mutationRate;

  vector<vector<int>> population;
  vector<double> fitness;

  vector<pair<int, int>> points = {{0, 0}, {1, 2}, {3, 1}, {5, 2},
                                   {7, 3}, {9, 1}, {8, 4}, {6, 5},
                                   {4, 5}, {2, 4}, {1, 6}, {3, 7}};
};

// ...

void testCalculateDistance() {
  const int numPoints = 12;
  vector<pair<int, int>> points = {{0, 0}, {1, 2}, {3, 1}, {5, 2},
                                   {7, 3}, {9, 1}, {8, 4}, {6, 5},
                                   {4, 5}, {2, 4}, {1, 6}, {3, 7}};
  vector<int> route = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0};

  TSPGeneticAlgorithm tspGA(numPoints, 0, 0, 0,
                            0);  // Dummy object to access calculateDistance
  double distance = tspGA.calculateDistance(route);

  // Assert that the distance is a non-negative value
  assert(distance >= 0);
}

void testSelectParentIndex() {
  const int populationSize = 120;
  TSPGeneticAlgorithm tspGA(0, populationSize, 0, 0,
                            0);  // Dummy object to access selectParentIndex
  vector<double> fitness(populationSize, 1.0);
  double totalFitness = accumulate(fitness.begin(), fitness.end(), 0.0);

  int parentIndex = tspGA.selectParentIndex(totalFitness);

  // Assert that the selected parent index is within the valid range
  assert(parentIndex >= 0 && parentIndex < populationSize);
}

void testCrossover() {
  const int numPoints = 12;
  vector<int> parent1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  vector<int> parent2 = {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  vector<int> child1(numPoints);
  vector<int> child2(numPoints);

  TSPGeneticAlgorithm tspGA(numPoints, 0, 0, 0,
                            0);  // Dummy object to access crossover
  tspGA.crossover(parent1, parent2, child1, child2);

  // Assert that the children are not equal to their respective parents
  assert(child1 != parent1 && child1 != parent2);
  assert(child2 != parent1 && child2 != parent2);
}

void runUnitTests() {
  testCalculateDistance();
  testSelectParentIndex();
  testCrossover();
  // Add more tests as needed
}

int main() {
  runUnitTests();

  const int numPoints = 12;
  const int populationSize = 120;
  const int numGenerations = 100;
  const double crossoverRate = 0.8;
  const double mutationRate = 0.02;

  TSPGeneticAlgorithm tspGA(numPoints, populationSize, numGenerations,
                            crossoverRate, mutationRate);
  tspGA.runAlgorithm();
  tspGA.printBestRoute();

  return 0;
}
