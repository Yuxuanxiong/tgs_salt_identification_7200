package training_process

import java.io.File
import java.util.{Calendar, Random}

import model_structure.UNET_RES
import org.apache.log4j.{BasicConfigurator, Logger}
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, ResizeImageTransform}
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.{FileStatsStorage, InMemoryStatsStorage}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
import org.nd4j.linalg.indexing.conditions.GreaterThan
import utils.IoUEvaluator

object UNET_RES_TRAIN extends App {

  protected val log : Logger = Logger.getLogger(UNET_RES_TRAIN.getClass)
  BasicConfigurator.configure()

  val height:Int=101
  val width:Int=101
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16
  val epochs:Int=1

  val unet : UNET_RES = UNET_RES(seed,channels,height,width)

  log.info("Loading data...")

  val mainPath:String = System.getProperty("user.dir")
  val imgPath:File = new File(mainPath, "Dataset/train/images")
  val maskPath:File = new File(mainPath, "Dataset/train/masks")
  val imgNum :Int = Option(imgPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)
  val maskNum:Int = Option(maskPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)

  if(imgNum != maskNum) throw new RuntimeException("Number of images and masks do not match! ")

  log.info(s"$imgNum images and masks found.")

  val imgSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))
  val maskSplit = new FileSplit(maskPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))

  val imgReader: ImageRecordReader = new ImageRecordReader(height, width, channels)
  val maskReader: ImageRecordReader = new ImageRecordReader(height,width, channels)

  val resize: ImageTransform = new ResizeImageTransform(width, height)

  imgReader.initialize(imgSplit, resize)
  maskReader.initialize(maskSplit)


  val scaler: DataNormalization = new ImagePreProcessingScaler(0,1)

  val imgIter = new RecordReaderDataSetIterator.Builder(imgReader, batchSize)
    .preProcessor(scaler)
    .build

  imgIter.next().getFeatures.shape()


  val maskIter = new RecordReaderDataSetIterator.Builder(maskReader, batchSize).build

  val net = unet.init_model
  log.info("Printing network summary...")
  println(net.summary)

//  setting up UI
  val uiServer:UIServer = UIServer.getInstance();

  val statsStorage :StatsStorage = new FileStatsStorage(new File(mainPath+"/target/UIStorageStats.dl4j"));
  val listenerFrequency = 1;
  net.setListeners(new StatsListener(statsStorage, listenerFrequency), new ScoreIterationListener(10));

  uiServer.attach(statsStorage);


  val ds:DataSet = new DataSet(imgIter.next(imgNum).getFeatures,
                                maskIter.next(maskNum).getFeatures.div(65535))

  ds.detach()

  val dsi:DataSetIterator = new SamplingDataSetIterator(ds, batchSize, imgNum)

  log.info(s"Start training...")
  for(i<-1 to epochs){
    log.info(s"Epoch $i, Time:${Calendar.getInstance.getTime}")
    net.fit(dsi)
  }

  log.info(s"Saving model...")
  val saveTo :File = new File(mainPath, "saved_models/UNET_RES.zip")
  saveTo.createNewFile()
  ModelSerializer.writeModel(net, saveTo, false)
  log.info("Finished")

//  val net = ModelSerializer.restoreComputationGraph(mainPath+"/Saved_Models/model_structure.UNET_RES.zip")

//  imgIter.reset()
//  maskIter.reset()
//
//  val cond1=new GreaterThan(0.5)
//
//  val out=net.output(imgIter.next.getFeatures)(0).cond(cond1)
//  val hat=maskIter.next.getFeatures.div(65535)
//  println("SampleOut: "+out)
//  println("SampleLabel: "+hat)
//
//  log.info("Evaluating...")
//
//  val eval:IoUEvaluator = IoUEvaluator(Seq(out), Seq(hat))
//  println("IoU: "+ eval.eval())


}
