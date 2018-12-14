import org.scalatest.{FlatSpec, Matchers}
import utils.DataImport


class DataImportSpec extends FlatSpec with Matchers{

  "Image size" should "101x101" in {

    val imageIter: DataImport = new DataImport("Dataset",101,101,1,12345,16,true)

    assertResult(imageIter.getImgIter().next().getFeatures.shape().toSeq.toString)("WrappedArray(16, 1, 101, 101)")
  }



}
