from pyspark import SparkContext, SparkConf
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local").setAppName("K_Means")
sc = SparkContext(conf=conf)


def kmeans(data, centroid, max_iterations, euclidean):
    iterCost = []

    distanceMeasure = ""
    if euclidean_distance:
        distanceMeasure = "Euclidean Distance"
    else:
        distanceMeasure = "Manhattan Distance"

    def closestCentroid(p):
        # print("inside function")
        closest = float('inf')
        bestIndex = 0
        dist = 0
        if euclidean:
            for i in range(len(centroid)):
                dist = np.sqrt(np.sum(np.subtract(p, centroid[i]) ** 2))
                if dist < closest:
                    closest = dist
                    bestIndex = i
            return (bestIndex, [p])
        else:
            for i in range(len(centroid)):
                dist = np.sum(abs(p - centroid[i]))
                if dist < closest:
                    closest = dist
                    bestIndex = i
            return (bestIndex, [p])

    def cost(p):
        closest = float('inf')
        cost = 0
        if euclidean:
            for i in range(len(centroid)):
                cost = np.sum(np.subtract(p,  centroid[i]) ** 2)
                if cost < closest:
                    closest = cost
            return closest
        else:
            for i in range(len(centroid)):
                cost = np.sum(abs(p - centroid[i]))
                if cost < closest:
                    closest = cost
            return closest

    def recomputeCentroids(p):
        temp = np.array(p)
        temp = np.mean(temp, axis=0)
        return temp

    for i in range(max_iterations):
        closestRDD = data.map(closestCentroid)
        costAll = data.map(cost).collect()
        totalCost = np.sum(costAll)
        reducedClosestRDD = closestRDD.reduceByKey(lambda a, b: a + b)
        newCentroidsRDD = reducedClosestRDD.map(lambda p: recomputeCentroids(p[1]))
        centroid = newCentroidsRDD.collect()
        iterCost.append(totalCost)

    print("Max Iterations: ", max_iterations, ", Distance Measure: ", distanceMeasure)
    for i in range(max_iterations):
        print("Iteration ", i + 1, " cost: ", iterCost[i])

    return iterCost


# Given Information
Iterations = 20
c1Path = "c1.txt"
c2Path = "c2.txt"
dataPath = "data.txt"

# Collecting and Converting data to float.
dataRDD = sc.textFile(dataPath).map(lambda line: np.array([float(x) for x in line.split(' ')]))
c1RDD = sc.textFile(c1Path).map(lambda c: np.array([float(x) for x in c.split(' ')]))
c2RDD = sc.textFile(c2Path).map(lambda c: np.array([float(x) for x in c.split(' ')]))

centroids = c1RDD.collect()
centroids2 = c2RDD.collect()

euclidean_distance = True
costC1Euclidean = kmeans(dataRDD, centroids, Iterations, euclidean_distance)
print("########################################################")
costC2Euclidean = kmeans(dataRDD, centroids2, Iterations, euclidean_distance)
print("########################################################")

euclidean_distance = False
costC1Manhattan = kmeans(dataRDD, centroids, Iterations, euclidean_distance)
print("########################################################")
costC2Manhattan = kmeans(dataRDD, centroids2, Iterations, euclidean_distance)
print("########################################################")

## Plotting Graph
XAxis = list(range(1, Iterations + 1))

plt.plot(XAxis, costC1Euclidean, label="c1")
plt.plot(XAxis, costC2Euclidean, label="c2")
plt.xlabel('Iterations')
plt.ylabel('Euclidean Cost')
plt.title('Euclidean')
plt.xticks(size=Iterations)
plt.legend()
plt.savefig("Euclidean.png")
plt.close()

# Manhattan cost plotting
plt.plot(XAxis, costC1Manhattan, label="c1")
plt.plot(XAxis, costC2Manhattan, label="c2")
plt.xlabel('Iterations')
plt.ylabel('Manhattan Cost')
plt.title('Manhattan')
plt.xticks(size=Iterations)
plt.legend()
plt.savefig("Manhattan.png")
plt.close()
