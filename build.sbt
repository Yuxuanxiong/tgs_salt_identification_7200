
name := "7200_project"

version := "1.0"

scalaVersion := "2.11.8"


assemblyJarName in assembly := "7200project.jar"

//mainClass in (Compile, packageBin) := Some("UNET_TRAIN_SPARK")


val dl4jVersion= "1.0.0-beta3"
val sparkVersion= "2.1.0"

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.apache.spark/spark-core

  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",

  // https://mvnrepository.com/artifact/org.apache.spark/spark-sql
//  "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
  
  // https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-client
//  "org.apache.hadoop" % "hadoop-client" % "2.2.0" % Provided,

  // https://mvnrepository.com/artifact/org.deeplearning4j/dl4j-spark
  ("org.deeplearning4j" %% "dl4j-spark" % s"${dl4jVersion}_spark_2")
    .exclude("org.apache.spark","spark-core"),
//    .exclude("org.deeplearning4j","deeplearning4j-ui-components"),

  // https://mvnrepository.com/artifact/org.deeplearning4j/dl4j-spark-parameterserver
  ("org.deeplearning4j" %% "dl4j-spark-parameterserver" % s"${dl4jVersion}_spark_2")
    .exclude("org.apache.spark", "spark-core")
    .exclude("org.deeplearning4j","dl4j-spark_2.11"),

  // https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-core
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,

  // https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-nn
//  "org.deeplearning4j" % "deeplearning4j-nn" % dl4jVersion,

  "org.nd4j" % "nd4j-native-platform" % dl4jVersion,
  
  // https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-ui
  "org.deeplearning4j" %% "deeplearning4j-ui" % dl4jVersion

)

assemblyMergeStrategy in assembly := {
//  case PathList("META-INF", "maven", xs @ _*) => MergeStrategy.last
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
//  case x if Assembly.isConfigFile(x) => MergeStrategy.concat
//  case PathList(ps @ _*) if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) => MergeStrategy.rename
//  case PathList("org","nd4j","serde",xs @ _*) => MergeStrategy.rename
//  case PathList("com","typesafe",xs @ _*) => MergeStrategy.first
//  case PathList("org","deeplearning4j",xs @ _*) => MergeStrategy.last
//  case PathList("org","nd4j",xs @ _*) => MergeStrategy.first
//  case PathList("org","apache","hadoop",xs @ _*) => MergeStrategy.last
//  case PathList("org","bytedeco","javacpp-presets",xs @ _*) => MergeStrategy.first
//  case PathList("commons-beanutils",xs @ _*) => MergeStrategy.last
//  case PathList("org","slf4j",xs @ _*) => MergeStrategy.first
//  case PathList("org","jetbrains",xs @ _*) => MergeStrategy.discard
//  case PathList("org","glassfish","hk2","external",xs @ _*) => MergeStrategy.first
//  case PathList("org","agrona",xs @ _*) => MergeStrategy.first
//  case a =>
//    val oldStrategy = (assemblyMergeStrategy in assembly).value
//    oldStrategy(a)
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

//assemblyShadeRules in assembly := Seq(
//  ShadeRule.rename("com.google.common.**" -> "repackaged.com.google.common.@1").inAll
//)
