package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs._
import loaders._
import preprocessing._

object CifarCreateLMDBApp {
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

    log("storing number of train minibatches in file")
    trainPartitionSizes.foreach(numTrainBatches => {
      val cifarDataFile = new File(sparkNetHome + "/infoFiles/cifar_10_num_train_batches.txt")
      cifarDataFile.getParentFile().mkdirs()
      val cifarDataWriter = new PrintWriter(cifarDataFile)
      cifarDataWriter.write(numTrainBatches.toString)
    })
    log("storing number of test minibatches in file")
    testPartitionSizes.foreach(numTestBatches => {
      val cifarDataFile = new File(sparkNetHome + "/infoFiles/cifar_10_num_test_batches.txt")
      cifarDataFile.getParentFile().mkdirs()
      val cifarDataWriter = new PrintWriter(cifarDataFile)
      cifarDataWriter.write(numTestBatches.toString)
    })

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

    log("create emtpy LMDBs on master")
    val LMDBCreator = new CreateLMDB(caffeLib)
    LMDBCreator.makeLMDB((new Array[(Array[ByteImage], Array[Int])](0)).iterator, sparkNetHome + "/caffe/examples/cifar10/cifar10_train_lmdb", height, width)
    LMDBCreator.makeLMDB((new Array[(Array[ByteImage], Array[Int])](0)).iterator, sparkNetHome + "/caffe/examples/cifar10/cifar10_test_lmdb", height, width)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    log("computing mean image")
    val meanImage = ComputeMean(trainMinibatchRDD, imShape, numTrainData.toInt)
    log("saving mean image on master")
    ComputeMean.writeMeanToBinaryProto(caffeLib, meanImage, sparkNetHome + "/caffe/examples/cifar10/mean.binaryproto")
    log("saving mean image on workers")
    workers.foreach( _ =>
      ComputeMean.writeMeanToBinaryProto(caffeLib, meanImage, sparkNetHome + "/caffe/examples/cifar10/mean.binaryproto")
    )

    log("finished creating databases")
  }
}
