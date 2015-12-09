package preprocessing

import com.sun.jna.Pointer

import org.apache.spark.rdd.RDD

import libs._

class CreateLMDB(caffeLib: CaffeLibrary) {
	val state = caffeLib.create_state()

  def makeLMDB(minibatchIt: Iterator[(Array[ByteImage], Array[Int])], dbName: String, height: Int, width: Int) = {
    caffeLib.create_db(state, dbName, dbName.length)
    var counter = 0
    val imBuffer = new Array[Byte](3 * height * width)
    while (minibatchIt.hasNext) {
      val (images, labels) = minibatchIt.next
      var i = 0
      while (i < images.length) {
        images(i).copyToBuffer(imBuffer)
        caffeLib.write_to_db(state, imBuffer, labels(i), height, width, counter.toString)
        counter += 1
        i += 1
      }
      caffeLib.commit_db_txn(state)
      print(counter.toString + " images written to db\n")
    }
		caffeLib.close_db(state)
  }
}
