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
  caffeLib.set_basepath(sparkNetHome + "/caffe/")

  // initialize nets on workers
  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt")
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

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // TODO: should get the size from the database instead of reading it from a file
    val testPartitionSizes = workers.map(_ => {
      val reader = new FileReader(new File(sparkNetHome + "/infoFiles/cifar_10_num_test_batches.txt"))
      var c = reader.read
      var size = 0
      while (c != -1) {
        size *= 10
        size += c.toChar.toString.toInt
        c = reader.read
      }
      size
    })
    val numTestMinibatches = testPartitionSizes.sum()

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      workers.foreach(_ => net.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        log("testing, i")
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
