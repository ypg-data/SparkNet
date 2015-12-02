package libs

class MinibatchSampler(minibatchIt: Iterator[(Array[ByteImage], Array[Int])], totalNumBatches: Int, numSampledBatches: Int) {
  // The purpose of this method is to take minibatchIt, which is an iterator
  // over images and labels, and to turn it into two iterators, one over images
  // and one over labels. The iterator over images is used to create a callback
  // that Caffe uses to get the next minibatch of images. The iterator over
  // labels is used to create a callback that Caffe uses to get the next
  // minibatch of labels. We cannot use the same iterator for both purposes
  // because incrementing the iterator in one callback will increment it in the
  // other callback as well, since they are the same object.

  // totalNumBatches = minibatchIt.length (but we can't call minibatchIt.length because that would consume the entire iterator)
  // numSampledBatches is the number of minibatches that we subsample from minibatchIt

  var it = minibatchIt // we need to update the iterator by calling it.drop, and we need it to be a var to do this
  val r = scala.util.Random
  val startIdx = r.nextInt(totalNumBatches - numSampledBatches + 1)
  val indices = Array.range(startIdx, startIdx + numSampledBatches)
  var indicesIndex = 0
  var currMinibatchPosition = -1

  val numMinibatchesToLoad = 20
  var imagePosition = 0
  var labelPosition = 0
  var loaded = false

  var currImageMinibatches = new Array[Array[ByteImage]](numMinibatchesToLoad)
  var currLabelMinibatches = new Array[Array[Int]](numMinibatchesToLoad)

  private def loadMinibatches() = {
    var i = 0
    while (i < numMinibatchesToLoad && indicesIndex < indices.length) {
      it = it.drop(indices(indicesIndex) - currMinibatchPosition - 1)
      currMinibatchPosition = indices(indicesIndex)
      indicesIndex += 1
      assert(it.hasNext)
      val (images, labels) = it.next
      currImageMinibatches(i) = images
      currLabelMinibatches(i) = labels
      i += 1
    }
  }

  def nextImageMinibatch(): Array[ByteImage] = {
    if (loaded == false) {
      loadMinibatches()
      loaded = true
    }
    val images = currImageMinibatches(imagePosition)
    imagePosition = (imagePosition + 1) % numMinibatchesToLoad
    if (imagePosition == 0 && labelPosition == 0) {
      loaded = false
    }
    return images
  }

  def nextLabelMinibatch(): Array[Int] = {
    if (loaded == false) {
      loadMinibatches()
      loaded = true
    }
    val labels = currLabelMinibatches(labelPosition)
    labelPosition = (labelPosition + 1) % numMinibatchesToLoad
    if (imagePosition == 0 && labelPosition == 0) {
      loaded = false
    }
    return labels
  }
}
