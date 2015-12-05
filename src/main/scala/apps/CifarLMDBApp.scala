package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs._
import loaders._
import preprocessing._

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
// TODO: for this to work, you need to create examples/cifar10/mean.binaryproto (and on all the workers)
// TODO: for this to work, we have to create the LMDB before we create the CaffeNet
// TODO: for this to work, we also need to copy some generic LMDB to the master

object CifarLMDBApp {
  val trainBatchSize = 100
  val testBatchSize = 100
  val channels = 3
  val width = 32
  val height = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  val sparkNetHome = "/root/SparkNet"
  System.load(sparkNetHome + "/build/libccaffe.so")
  val caffeLib = CaffeLibrary.INSTANCE
  caffeLib.set_basepath(sparkNetHome + "/caffe/") // TODO: careful about logging with relative file paths, because this calls chdir

  // initialize nets on workers
  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt")
  //netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, width)
  val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_solver.prototxt", netParameter, None)
  val net = CaffeNet(caffeLib, solverParameter)

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("CifarLMDB")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    // information for logging
    val startTime = System.currentTimeMillis()
    val trainingLog = new PrintWriter(new File(sparkNetHome + "/training_log_" + startTime.toString + ".txt" ))
    def log(message: String, i: Int = -1) {
      val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
      if (i == -1) {
        trainingLog.write(elapsedTime.toString + ": "  + message + "\n")
      } else {
        trainingLog.write(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
      }
      trainingLog.flush()
    }

    var netWeights = net.getWeights()

    val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")
    log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    log("loading test data")
    var testRDD = sc.parallelize(loader.testImages.zip(loader.testLabels))

    log("repartition data")
    trainRDD = trainRDD.repartition(numWorkers)
    testRDD = testRDD.repartition(numWorkers)

    log("processing train data")
    val trainConverter = new ScaleAndConvert(trainBatchSize, height, width)
    var trainMinibatchRDD = trainConverter.makeMinibatchRDDWithoutCompression(trainRDD).persist()
    val numTrainMinibatches = trainMinibatchRDD.count()
    log("numTrainMinibatches = " + numTrainMinibatches.toString)

    log("processing test data")
    val testConverter = new ScaleAndConvert(testBatchSize, height, width)
    var testMinibatchRDD = testConverter.makeMinibatchRDDWithoutCompression(testRDD).persist()
    val numTestMinibatches = testMinibatchRDD.count()
    log("numTestMinibatches = " + numTestMinibatches.toString)

    val numTrainData = numTrainMinibatches * trainBatchSize
    val numTestData = numTestMinibatches * testBatchSize

    val trainPartitionSizes = trainMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testMinibatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    /*
    log("write train data to LMDB")
    trainMinibatchRDD.mapPartitions(minibatchIt => {
      val LMDBCreator = new CreateLMDB(caffeLib)
      LMDBCreator.makeLMDB(minibatchIt, sparkNetHome + "/caffe/examples/cifar10/cifar10_train_lmdb", height, width)
      Array(0).iterator
    }).foreach(_ => ())

    log("write test data to LMDB")
    testMinibatchRDD.mapPartitions(minibatchIt => {
      val LMDBCreator = new CreateLMDB(caffeLib)
      LMDBCreator.makeLMDB(minibatchIt, sparkNetHome + "/caffe/examples/cifar10/cifar10_test_lmdb", height, width)
      Array(0).iterator
    }).foreach(_ => ())
    */

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      workers.foreach(_ => net.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        log("testing, i")
        // TODO: actually need to tell Caffe the number of iterations of testing
        val testScores = testPartitionSizes.map(
          size => {
            net.setNumTestBatches(size)
            net.test()
          }
        ).cache()
        val testScoresAggregate = testScores.reduce((a, b) => (a, b).zipped.map(_ + _))
        val accuracies = testScoresAggregate.map(v => 100F * v / numTestMinibatches)
        log("%.2f".format(accuracies(0)) + "% accuracy", i)
      }

      log("training", i)
      val syncInterval = 10
      workers.foreach(_ => net.train(syncInterval))

      log("collecting weights", i)
      netWeights = workers.map(_ => { net.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      i += 1
    }

    log("finished training")
  }
}
