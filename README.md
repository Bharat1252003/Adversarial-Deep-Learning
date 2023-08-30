# Adversarial-Deep-Learning
Adversial learning for image classification using game theory

The algorithm uses the genetic algorithm and an original CNN to select and populate a selection of image alterations which are then used to make adverarial examples. Using these adversarial examples, a new CNN is trained. The new CNN is used again with the genetic algorithm to generate more adversarial examples. The Stackelberg game is expected to converge to a robust CNN that is the most optimum to resist adversarial attacks using the genetic algorithm. The implementation is based on the paper A. S. Chivukula and W. Liu, "Adversarial Deep Learning Models with Multiple Adversaries," in IEEE Transactions on Knowledge and Data Engineering, vol. 31, no. 6, pp. 1066-1079, 1 June 2019, doi: 10.1109/TKDE.2018.2851247.

Change save locations in structure.py, GA.py