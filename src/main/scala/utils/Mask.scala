package utils


case class Mask(v:String) {

  def apply(v:String):Mask = v match {
    case "" => null
    case v:String => if(v.length%2!=0) new Mask(v) else throw new Exception("Invalid length of utils.Mask string. should be even")
    case _ => throw new Exception("Error parsing utils.Mask string")
  }

  lazy val tupleVec = Mask.toTupleVec(v)

  lazy val pixels = Mask.toPixels(tupleVec)

//  lazy val matrix = utils.Mask.toMatrix(pixels)

}


object Mask {

  val fixedImageH=101
  val fixedImageW=101


  def toTupleVec(v:String):Vector[(Int,Int)] = {
    //parse the string to a vec of Int
    val temp:Vector[Int] = v.split(" ").map(_.toInt).toVector
    //pair all elements into tuples
    temp.foldLeft(Vector.empty[(Int,Int)]) ((result,next)=>{
      if(result.nonEmpty && result.last._2==0) {result.dropRight(1) :+ (result.last._1,next)}
      else {result :+ (next,0)}
    })
  }

  def toPixels(v:Vector[(Int,Int)]): Vector[Int] = {
    //parse the TupleVectors into pixel indexes
    v.flatMap((t:(Int,Int)) => for(i<-0 until t._2) yield t._1+i)
  }


//  def toMatrix(v:Vector[Int]):DenseMatrix = DenseMatrix(fixedImageH, fixedImageW, v.map(_.toDouble).toArray)

}

