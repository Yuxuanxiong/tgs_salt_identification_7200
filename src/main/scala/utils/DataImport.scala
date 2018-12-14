package utils

import java.io.File
import java.util.Random
import org.apache.log4j.{Logger, BasicConfigurator}
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, PipelineImageTransform, ResizeImageTransform, ShowImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}

case class DataImport(dataSetPath:String,
                      height:Int,
                      width:Int,
                      channels:Int,
                      seed:Long,
                      batchSize:Int,
                      showBefore:Boolean) {
//
//  val height: Int = 101
//  val width: Int = 101
//  val channels: Int = 1
//  val seed: Long = 12345
//  val batchSize: Int = 16

  val log : Logger = Logger.getLogger(this.getClass)
  BasicConfigurator.configure()

  val imgPath: File = new File(dataSetPath, "train/images")
  val maskPath: File = new File(dataSetPath, "train/masks")
  val imgNum: Int = Option(imgPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)
  val maskNum: Int = Option(maskPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)

  if(imgNum != maskNum) throw new RuntimeException("Number of images and masks do not match! ")

  log.info(s"$imgNum images and masks found.")

  val imgSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))
  val maskSplit = new FileSplit(maskPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))

  val imgReader: ImageRecordReader = new ImageRecordReader(height, width, channels)
  val maskReader: ImageRecordReader = new ImageRecordReader(height, width, channels)

  val resize: ImageTransform = new ResizeImageTransform(width, height)
  val show: ShowImageTransform = new ShowImageTransform("show before")
  val pipeline: PipelineImageTransform = new PipelineImageTransform(resize, show)

  if(showBefore) imgReader.initialize(imgSplit, pipeline)
  else imgReader.initialize(imgSplit, resize)
  maskReader.initialize(maskSplit)
  val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)

  def getImgIter()= new RecordReaderDataSetIterator.Builder(imgReader, batchSize).preProcessor(scaler).build
  def getMaskIter() = new RecordReaderDataSetIterator.Builder(maskReader, batchSize).build

}
