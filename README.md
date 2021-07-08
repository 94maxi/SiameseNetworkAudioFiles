# SiameseNetworkAudioFiles\
This program first takes audio files and turns them into Mel Spectograms
Those Mel Spectograms are then Used to Train a Siamese Network.
A distance Matrix is then created.
For a given distance matrix and labels of all samples, this function calculates two performance measures:
1) The mean CMC scores for n = [1, 3, 5, 10].
2) A mean accuracy metric. This metric calculates how many of the k samples that belong to the same class are among the first k ranked elements.
       
